from langchain_core.messages import ToolMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from graph_chat.agent_assistant import update_flight_runnable, update_flight_sensitive_tools, update_flight_safe_tools, \
    book_car_rental_runnable, book_car_rental_safe_tools, book_car_rental_sensitive_tools, book_hotel_runnable, \
    book_hotel_safe_tools, book_hotel_sensitive_tools, book_excursion_runnable, book_excursion_safe_tools, \
    book_excursion_sensitive_tools
from graph_chat.assistant import CtripAssistant
from graph_chat.base_data_model import CompleteOrEscalate
from graph_chat.entry_node import create_entry_node
from tools.tools_handler import create_tool_node_with_fallback


# 航班助手的子工作流 —— 传入主工作流，传回添加了子工作流后的主工作流
def build_flight_graph(builder: StateGraph) -> StateGraph:
    # 更新航班的入口节点 —— 一个过渡节点，在实际执行操作之前能更清晰地提示模型自己的身份与任务
    # 注意，add_node 的参数是节点名与要执行的函数，但不是返回函数的实例而是返回函数本身，即不加括号！
    # 之所以这里有括号且传了参数，是因为是一个闭包的写法，其内部返回的函数是函数对象本身！
    # 这个闭包写法相当于：
    # builder.add_node(
    #     "enter_update_flight",
    #     entry_node # 明面上的函数返回的真正函数，框架会自动传递state给它，它负责输出更新state的字典
    #     而由于它被定义在外层函数里面，因此它可以接收外层函数接受的参数
    #     之所以要用这种函数相互嵌套的闭包写法，是因为不同的助手节点创造的节点信息也是不同的
    #     因此要把参数传给不同的助手，而add_node又规定其中的函数能且仅能以唯一的state作为输入，因此只能使用闭包
    #     闭包达到的效果是，实际执行时传入的参数只有state，但传给外部函数的参数也可以被使用，达到了传入三个参数的效果
    # )
    builder.add_node(
        "enter_update_flight",
        # 这里实际上是一个闭包写法，创建入口节点，指定助理名称和新对话状态
        create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
    )

    # 添加处理航班更新的实际节点
    # 这里的 update_flight_runnable 是一个定义好的，绑定了工具（查询、修改、取消航班）并设好了提示词（告知模型自己负责更新航班）的模型
    # 这里是 add_node 的类写法，即传参是不再是传入一个函数对象，而是传入一个类实例 —— 特别注意，函数不传实例传对象，类则传实例！
    # 因此，CtripAssistant(update_flight_runnable)就是一个类实例，通过 __init__ 得到
    # 在 Python 中，对于类 x，x()就得到了一个类实例，而这个过程会先调用默认的new方法，创建类实例，再调用init方法，给类的属性赋值
    # 此时，这个类还应具有call方法，即能够被当成函数使用，其返回的也是关于state的更新
    builder.add_node("update_flight", CtripAssistant(update_flight_runnable))
    # 连接入口节点到实际处理节点（这样的边在绘图中体现为实线边，唯一且一定会执行）
    builder.add_edge("enter_update_flight", "update_flight")

    # 添加敏感工具和安全工具的节点
    # 这里是add_node的另一种写法：既不是传递函数对象，也不是传递类实例，而是传递一个函数实例，这个函数实例返回了一个类实例
    # 也就是说，这种写法间接地相当于传递了类实例
    # 这个类实例实现的功能为，当发生错误时，返回对应的，要更新的state（在message中提示发生错误）
    builder.add_node(
        "update_flight_sensitive_tools",
        create_tool_node_with_fallback(update_flight_sensitive_tools),  # 敏感工具节点，包含可能修改数据的操作
    )
    builder.add_node(
        "update_flight_safe_tools",
        create_tool_node_with_fallback(update_flight_safe_tools),  # 安全工具节点，通常只读查询
    )

    def route_update_flight(state: dict):
        """
        根据当前状态路由航班更新流程。

        :param state: 当前对话状态字典
        :return: 下一步应跳转到的节点名
        """
        # 如果返回的消息中没有工具调用，说明该结束了（无论是更换什么节点，都需要调用工具，没调用就说明结束了）
        route = tools_condition(state)
        if route == END:
            return END  # 无工具调用则直接结束
        tool_calls = state["messages"][-1].tool_calls  # 获取最后一条消息中的工具调用
        # 最近的记录中，是否至少有一个调用了 CompleteOrEscalate
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            return "leave_skill"  # 如果用户请求取消或退出，则跳转至leave_skill节点
        safe_tool_names = [t.name for t in update_flight_safe_tools]  # 获取所有安全工具的名字
        if all(tc["name"] in safe_tool_names for tc in tool_calls):  # 如果所有调用的工具都是安全工具
            return "update_flight_safe_tools"  # 跳转至安全工具处理节点
        return "update_flight_sensitive_tools"  # 否则跳转至敏感工具处理节点

    # 添加边，连接敏感工具和安全工具节点回到航班更新处理节点
    builder.add_edge("update_flight_sensitive_tools", "update_flight")
    builder.add_edge("update_flight_safe_tools", "update_flight")

    # 根据条件路由航班更新流程
    builder.add_conditional_edges(
        "update_flight",
        route_update_flight,
        ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END], # 下一个可能的节点
    )

    # 此节点将用于所有子助理的退出
    def pop_dialog_state(state: dict) -> dict:
        """
        弹出对话栈并返回主助理。
        这使得完整的图可以明确跟踪对话流，并根据需要委托控制给特定的子图。
        :param state: 当前对话状态字典
        :return: 包含新的对话状态和消息的字典
        """
        messages = []
        # 如果上一条消息调用了工具，则将其基本信息送入message中，并对state进行更新
        if state["messages"][-1].tool_calls:
            # 注意：目前不处理LLM同时执行多个工具调用的情况
            messages.append(
                ToolMessage(
                    content="正在恢复与主助理的对话。请回顾之前的对话并根据需要协助用户。",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            )
        return {
            # 更新对话状态为弹出 —— 这里的 pop不是具体值，而是表示要弹出（被识别为一个“命令”）
            # 具体的值在state中定义，只能是那五个
            "dialog_state": "pop",
            "messages": messages,  # 返回消息列表
        }

    # 添加退出技能节点，并连接回主助理
    builder.add_node("leave_skill", pop_dialog_state)
    builder.add_edge("leave_skill", "primary_assistant")
    return builder


def build_car_graph(builder: StateGraph) -> StateGraph:
    # 租车助理 的子工作流
    # 添加入口节点，当需要预订租车时使用
    builder.add_node(
        "enter_book_car_rental",
        create_entry_node("Car Rental Assistant", "book_car_rental"),  # 创建入口节点，指定助理名称和新对话状态
    )
    builder.add_node("book_car_rental", CtripAssistant(book_car_rental_runnable))  # 添加处理租车预订的实际节点
    builder.add_edge("enter_book_car_rental", "book_car_rental")  # 连接入口节点到实际处理节点

    # 添加安全工具和敏感工具的节点
    builder.add_node(
        "book_car_rental_safe_tools",
        create_tool_node_with_fallback(book_car_rental_safe_tools),  # 安全工具节点，通常只读查询
    )
    builder.add_node(
        "book_car_rental_sensitive_tools",
        create_tool_node_with_fallback(book_car_rental_sensitive_tools),  # 敏感工具节点，包含可能修改数据的操作
    )

    def route_book_car_rental(state: dict):
        """
        根据当前状态路由租车预订流程。

        :param state: 当前对话状态字典
        :return: 下一步应跳转到的节点名
        """
        route = tools_condition(state)  # 判断下一步的方向
        if route == END:
            return END  # 如果结束条件满足，则返回END
        tool_calls = state["messages"][-1].tool_calls  # 获取最后一条消息中的工具调用
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)  # 检查是否调用了CompleteOrEscalate
        if did_cancel:
            return "leave_skill"  # 如果用户请求取消或退出，则跳转至leave_skill节点
        safe_toolnames = [t.name for t in book_car_rental_safe_tools]  # 获取所有安全工具的名字
        if all(tc["name"] in safe_toolnames for tc in tool_calls):  # 如果所有调用的工具都是安全工具
            return "book_car_rental_safe_tools"  # 跳转至安全工具处理节点
        return "book_car_rental_sensitive_tools"  # 否则跳转至敏感工具处理节点

    # 添加边，连接敏感工具和安全工具节点回到租车预订处理节点
    builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
    builder.add_edge("book_car_rental_safe_tools", "book_car_rental")

    # 根据条件路由租车预订流程
    builder.add_conditional_edges(
        "book_car_rental",
        route_book_car_rental,
        [
            "book_car_rental_safe_tools",
            "book_car_rental_sensitive_tools",
            "leave_skill",
            END,
        ],
    )
    return builder


# 酒店预订助理
def builder_hotel_graph(builder: StateGraph) -> StateGraph:
    # 添加入口节点，当需要预订酒店时使用
    builder.add_node(
        "enter_book_hotel",
        create_entry_node("酒店预订助理", "book_hotel"),  # 创建入口节点，指定助理名称和新对话状态
    )
    builder.add_node("book_hotel", CtripAssistant(book_hotel_runnable))  # 添加处理酒店预订的实际节点
    builder.add_edge("enter_book_hotel", "book_hotel")  # 连接入口节点到实际处理节点

    # 添加安全工具和敏感工具的节点
    builder.add_node(
        "book_hotel_safe_tools",
        create_tool_node_with_fallback(book_hotel_safe_tools),  # 安全工具节点，通常只读查询
    )
    builder.add_node(
        "book_hotel_sensitive_tools",
        create_tool_node_with_fallback(book_hotel_sensitive_tools),  # 敏感工具节点，包含可能修改数据的操作
    )

    def route_book_hotel(state: dict):
        """
        根据当前状态路由酒店预订流程。

        :param state: 当前对话状态字典
        :return: 下一步应跳转到的节点名
        """
        route = tools_condition(state)  # 判断下一步的方向
        if route == END:
            return END  # 如果结束条件满足，则返回END
        tool_calls = state["messages"][-1].tool_calls  # 获取最后一条消息中的工具调用
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)  # 检查是否调用了CompleteOrEscalate
        if did_cancel:
            return "leave_skill"  # 如果用户请求取消或退出，则跳转至leave_skill节点
        safe_toolnames = [t.name for t in book_hotel_safe_tools]  # 获取所有安全工具的名字
        if all(tc["name"] in safe_toolnames for tc in tool_calls):  # 如果所有调用的工具都是安全工具
            return "book_hotel_safe_tools"  # 跳转至安全工具处理节点
        return "book_hotel_sensitive_tools"  # 否则跳转至敏感工具处理节点

    # 添加边，连接敏感工具和安全工具节点回到酒店预订处理节点
    builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
    builder.add_edge("book_hotel_safe_tools", "book_hotel")

    # 根据条件路由酒店预订流程
    builder.add_conditional_edges(
        "book_hotel",
        route_book_hotel,
        ["leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", END],
    )
    return builder


# 构建一个旅游产品的子图
def builder_excursion_graph(builder: StateGraph) -> StateGraph:
    # 添加入口节点，当需要预订游览或获取旅行推荐时使用
    builder.add_node(
        "enter_book_excursion",
        create_entry_node("旅行推荐助理", "book_excursion"),  # 创建入口节点，指定助理名称和新对话状态
    )
    builder.add_node("book_excursion", CtripAssistant(book_excursion_runnable))  # 添加处理游览预订的实际节点
    builder.add_edge("enter_book_excursion", "book_excursion")  # 连接入口节点到实际处理节点

    # 添加安全工具和敏感工具的节点
    builder.add_node(
        "book_excursion_safe_tools",
        create_tool_node_with_fallback(book_excursion_safe_tools),  # 安全工具节点，通常只读查询
    )
    builder.add_node(
        "book_excursion_sensitive_tools",
        create_tool_node_with_fallback(book_excursion_sensitive_tools),  # 敏感工具节点，包含可能修改数据的操作
    )

    def route_book_excursion(state: dict):
        """
        根据当前状态路由游览预订流程。

        :param state: 当前对话状态字典
        :return: 下一步应跳转到的节点名
        """
        route = tools_condition(state)  # 判断下一步的方向
        if route == END:
            return END  # 如果结束条件满足，则返回END
        tool_calls = state["messages"][-1].tool_calls  # 获取最后一条消息中的工具调用
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)  # 检查是否调用了CompleteOrEscalate
        if did_cancel:
            return "leave_skill"  # 如果用户请求取消或退出，则跳转至leave_skill节点
        safe_toolnames = [t.name for t in book_excursion_safe_tools]  # 获取所有安全工具的名字
        if all(tc["name"] in safe_toolnames for tc in tool_calls):  # 如果所有调用的工具都是安全工具
            return "book_excursion_safe_tools"  # 跳转至安全工具处理节点
        return "book_excursion_sensitive_tools"  # 否则跳转至敏感工具处理节点

    # 添加边，连接敏感工具和安全工具节点回到游览预订处理节点
    builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
    builder.add_edge("book_excursion_safe_tools", "book_excursion")

    # 根据条件路由游览预订流程
    builder.add_conditional_edges(
        "book_excursion",
        route_book_excursion,
        ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END],
    )
    return builder
