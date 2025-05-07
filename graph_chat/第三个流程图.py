import uuid
import os

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from graph_chat.assistant import CtripAssistant, assistant_runnable, primary_assistant_tools
from graph_chat.base_data_model import ToFlightBookingAssistant, ToBookCarRental, ToHotelBookingAssistant, \
    ToBookExcursion
from graph_chat.build_child_graph import build_flight_graph, builder_hotel_graph, build_car_graph, \
    builder_excursion_graph
from tools.flights_tools import fetch_user_flight_information
from graph_chat.draw_png import draw_graph
from graph_chat.state import State
from tools.init_db import update_dates
from tools.tools_handler import create_tool_node_with_fallback, _print_event


# 定义了一个流程图的构建对象
builder = StateGraph(State)
# state是自定义的，包括：
# messages：一个列表，用于存储所有历史记录
# user_info：一个字符串，存储用户的个人信息
# dialog_state：一个字符串，存储当前的助手身份

def get_user_info(state: State):
    """
    获取用户的航班信息并更新状态字典。
    参数:
        state (State): 当前状态字典。
    返回:
        dict: 包含用户信息的新状态字典。
    """
    # fetch_user_flight_information中定义了函数传入的必须是RunnableConfig（官方库，是一个存储配置信息的字典）
    return {"user_info": fetch_user_flight_information.invoke({})}

# 新增：fetch_user_info节点首先运行，这意味着我们的助手可以在不采取任何行动的情况下看到用户的航班信息
# 节点：需要定义名称以及要执行的函数，函数的输入必须仅为state，且会由框架默认输入；函数的输出为对state的更新（字典形式）
# 这里为什么get_user_info后面没有括号：add_node只是创建节点，指明节点的名字与运行时的操作，但这只是初始化而不是实际执行！
# 因此这里是把函数本身作为参数传入，指明节点的操作；而不是加括号，调用函数运行的结果
builder.add_node('fetch_user_info', get_user_info)
builder.add_edge(START, 'fetch_user_info')

# 添加 四个业务助理的子工作流
builder = build_flight_graph(builder)
builder = builder_hotel_graph(builder)
builder = build_car_graph(builder)
builder = builder_excursion_graph(builder)

# 添加主助理
# 类写法，assistant_runnable是用来初始化类的，包括了llm，提示词与能使用的工具
# 其中，工具包括政策查询工具，网络搜索工具与搜索航班的工具（功能型），还包括转向各个子助理的工具（信号型）
# 信号型工具中明确了需要跳转到哪个子助手，并在其中定义了需要自助手接受的信息，这些信息会通过state的message传给子助手
builder.add_node('primary_assistant', CtripAssistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)  # 主助理工具节点，包含各种工具
)

def route_primary_assistant(state: dict):
    """
    根据当前状态判断路由到子助手节点。
    :param state: 当前对话状态字典
    :return: 下一步应跳转到的节点名
    """
    route = tools_condition(state)  # 判断下一步的方向
    if route == END:
        return END  # 如果结束条件满足，则返回END
    tool_calls = state["messages"][-1].tool_calls  # 获取最后一条消息中的工具调用
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"  # 跳转至航班预订入口节点
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"  # 跳转至租车预订入口节点
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"  # 跳转至酒店预订入口节点
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"  # 跳转至游览预订入口节点
        return "primary_assistant_tools"  # 否则跳转至主助理工具节点
    raise ValueError("无效的路由")  # 如果没有找到合适的工具调用，抛出异常

# 条件边：符合谁的条件就跳转到谁（取决于之前对话对state的更新）
builder.add_conditional_edges(
    'primary_assistant',
    route_primary_assistant,
    # path_map的作用是，限定返回的值必须在以下值之中，否则报错
    [
        "enter_update_flight",  # 航班 子助手的入口节点
        "enter_book_car_rental",  # 租车 子助手的入口节点
        "enter_book_hotel",   # 酒店 子助手的入口节点
        "enter_book_excursion",   # 旅游景点 子助手的入口节点
        "primary_assistant_tools",  # 主助手的工具： 全网搜索工具，查询企业政策的工具
        END,
    ]
)

# 添加边：调用工具后，返回主助手
builder.add_edge('primary_assistant_tools', 'primary_assistant')

# 每个委托的工作流可以直接响应用户。当用户响应时，我们希望返回到当前激活的工作流
def route_to_workflow(state: dict) -> str:
    """
    如果我们在一个委托的状态中，直接路由到相应的助理。
    :param state: 当前对话状态字典
    :return: 应跳转到的节点名
    """
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"  # 如果没有对话状态，返回主助理
    return dialog_state[-1]  # 返回最后一个对话状态

# 没有加path_map的限定，因为state中的dialog_state已经经过了严格的限定（只能是五个选项之一），因此不用担心值出错
builder.add_conditional_edges("fetch_user_info", route_to_workflow)  # 根据获取用户信息进行路由

# 实例化一个用于保存/恢复内存状态的对象
memory = MemorySaver()

graph = builder.compile(
    # 检查点：如果工作流中发生中断或失败，memory 将用于恢复工作流的状态。
    checkpointer=memory,
    # 工作流执行到这些节点时会中断，并向用户确认
    # 中断意味着流的停止，且由于这是一个人为造成的中断，模型仍然可以基于原本的定义得知其“如果不中断的话，下一个节点是什么”
    # 此时current_state.next就会为true
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ]
)

# draw_graph(graph, 'graph4.png')

# 生成随机的唯一会话id
session_id = str(uuid.uuid4())
update_dates()  # 每次测试的时候：保证数据库是全新的，保证，时间也是最近的时间

# 配置参数，包含乘客ID和线程ID
config = {
    "configurable": {
        # passenger_id用于我们的航班工具，以获取用户的航班信息
        "passenger_id": "3442 587242",
        # 检查点由session_id访问
        "thread_id": session_id,
    }
}

_printed = set()  # set集合，避免重复打印

# 执行工作流
while True:
    question = input('用户：')
    # 退出逻辑，目前只是样本，当用户输入的单词包括 q/exit/quit 时退出，也没有进行中译英
    if question.lower() in ['q', 'exit', 'quit']:
        print('对话结束，拜拜！')
        break
    else:
        # 参数：input——对state的初始化更新，config——之前动态定义的配置字典，stream_mode——返回值（events）的格式，具体如下：
        # "values"只返回最终状态值（最常用）
        # "messages"返回 LangGraph 中间所有消息
        # "all"	返回执行 trace，包括每个节点的日志记录等
        events = graph.stream({'messages': ('user', question)}, config, stream_mode='values')
        # 打印消息，直到中断发生（或者用户退出退出）——builder.compile中定义了，当涉及到敏感工具时就会中断
        for event in events:
            _print_event(event, _printed)

        # 中断发生
        current_state = graph.get_state(config)
        # 此时虽然中断了，但模型仍然会判断得到一个“本应执行的下一个节点”，即current_state.next就会为true
        if current_state.next:
            user_input = input(
                "您是否批准上述操作？输入'y'继续；否则，请说明您请求的更改。\n"
            )
            if user_input.strip().lower() == "y":
                # 之前的流中断了，因此要继续执行 —— 由于之前只是中断，因此会自动继续利用之前的state，继续流程
                events = graph.stream(None, config, stream_mode='values')
                # 打印消息
                for event in events:
                    _print_event(event, _printed)
            else:
                # 如果拒绝，继续流程的同时要在message中明确指明工具被拒绝以及拒绝的原因，方便模型进行后续的处理
                result = graph.stream(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"Tool的调用被用户拒绝。原因：'{user_input}'。",
                            )
                        ]
                    },
                    config,
                )
                # 打印事件详情
                for event in result:
                    _print_event(event, _printed)
