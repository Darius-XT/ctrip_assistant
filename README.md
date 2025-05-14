## 项目简介:

一个基于 LangGraph 实现的智能多代理旅行助手系统，通过主助理与多个专业子助理（航班、酒店、租车、游览）的协作，自动化处理复杂的旅行预订、查询与管理任务

## 主要工作：

- 设计主助理与专业子助理协同工作模式，有效分解并处理复杂的复合型旅行任务
- 主助理能基于 LLM 理解，自动将具体任务分配给具备相应能力的子助理
- 通过 LangGraph 精确管理对话状态，支持跨多轮、跨助理切换的复杂、连贯交互
- 集成数据库操作、外部 API 调用、RAG 及流程控制信号工具，显著增强 Agent 的问题解决能力和自主性
- 对关键操作引入用户确认环节，结合错误处理机制，提升系统可靠性和用户体验

## 项目结构：

![graph8](https://github.com/user-attachments/assets/43465bec-b382-4087-94c5-2ecefd3d47b6)

## 项目细节

定义的state类:
包括LangChain官方的AnyMessage, 一个自定义的user_info字符串(用于存储用户的信息), 以及一个指示当前助手类型的dialog_state(==注意它不是一个字符串而是一个字符串栈!==)

```python
class State(TypedDict):
    """
    定义一个结构化的字典类型，用于存储对话状态信息。
    字段:
        messages (list[AnyMessage]):
        AnyMessage = Union[
        HumanMessage：用户发来的消息
        AIMessage：AI 模型回复的消息
        SystemMessage：系统层级的消息（控制提示）
        ToolMessage：工具返回的消息
        ToolCall：模型请求调用某个工具
        FunctionMessage：函数调用（Function Calling API）返回的消息
        ]
        user_info (str): 自定义的，存储用户信息的字符串，例如机票、酒店等的信息
        dialog_state (list[Literal["assistant", "update_flight", "book_car_rental",
                                    "book_hotel", "book_excursion"]]): 指示当前对话是跟哪个助手进行的
    """
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[  # 其元素严格限定为上述五个字符串值之一。这种做法确保了对话状态管理逻辑的一致性和正确性，避免了意外的状态值导致的潜在问题。
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]
```

定义的config:
包括乘客id和线程id

```python
# 配置参数，包含乘客ID和线程ID
config = {
    "configurable": {
        # passenger_id用于我们的航班工具，以获取用户的航班信息
        "passenger_id": "3442 587242",
        # 检查点由session_id访问
        "thread_id": session_id,
    }
}
```

各助理的介绍:

1. 主助理, 负责将任务分配给不同的子助理
2. 航班子助理
3. 酒店子助理
4. 租车子助理
5. 游览子助理

## 项目流程

### 定义部分

#### 主入口节点(fetch_user_info)

首先由Start节点进入主入口(fetch_user_info), 对应执行的函数为查询数据库中, 与用户航班相关的一切信息, 并将结果存入state的user_info中, 结果格式如下:

```python
{
    "ticket_no": "ABC1234567",
    "book_ref": "BR0001",
    "flight_id": "987",
    "flight_no": "UA101",
    "departure_airport": "JFK",
    "arrival_airport": "LAX",
    "scheduled_departure": "2025-06-01 08:00:00",
    "scheduled_arrival": "2025-06-01 11:30:00",
    "seat_no": "12A",
    "fare_conditions": "Economy"
}
```

#### 子助理入口节点(enter_update_flight)

- 接下来, 对每个子助理, 构建入口节点 enter_up_flight(以机票子助理为例)
- 入口节点负责提示当前的助理自己的身份, 以及自己的任务, 注意==提示是放在tool_message中的, 以防止对系统与用户消息的污染==
- 四个子助理都要创建入口节点, 因此存在大量可复用的部分; 然而LangChain的绑定在node中的函数要求是必须仅以state作为输入, 因此==使用闭包, 达到"实际上传了多个参数"的效果==

#### 子助理更新节点(update_flight): 仍以机票助手为例==(核心)==

- 这个node负责更新航班相关信息, ==准确地说, 是"判断要调用哪些工具, 进行哪些更新", 但并不能实际进行这些更新!==
- node绑定的不再是一个函数, 而是一个具有call方法的类(允许将其当作函数使用, 适用于更复杂的，需要进行更多定制的场景
- 这里的函数以一个runnable对象作为init的输入, 并将其保存在self的参数中, 再在call方法中使用(call方法的传参仍然必须是规定格式, 即只能传入state和config)
- 这里的runnable对象实际上就是一个负责更新节点的LLM, 通过LangChain绑定了对应的提示词和工具 
  - 其提示词的功能为: 告知模型自己负责更新机票相关的操作, 并且把之前存在user_info中的机票相关信息传给模型
  - 绑定的工具包括:
    - ==安全工具(无需用户确认: 搜索航班)==
    - ==敏感工具(需要用户确认: 改签航班 & 取消航班)==
    - ==CompleteOrEscalate工具: 提示子助理什么时候应该返回主助理==
      - 属性
        - cancel: 是否离开当前身份, 返回主助理
        - reason: 这样做的理由
      - 示例(通过Config类)

```python
class CompleteOrEscalate(BaseModel):  # 定义数据模型类 —— 只要继承了BaseModel，就可以被当做工具使用（信号型）
    """
    一个工具，用于标记当前任务为已完成和/或将对话的控制权升级到主助理，
    主助理可以根据用户的需求重新路由对话。
    """

    cancel: bool = True  # 默认取消任务
    reason: str  # 取消或升级的原因说明

    # 示例：分别展示了可能会调用类的三种情况 —— 任务完成，用户意图变更与权限不足、需要调用其它工具
    class Config:  # 内部类 Config: json_schema_extra: 这个字段包含了一些示例数据
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "用户改变了对当前任务的想法。",
            },
            "example2": {
                "cancel": True,
                "reason": "我已经完成了任务。",
            },
            "example3": {
                "cancel": False,
                "reason": "我需要搜索用户的电子邮件或日历以获取更多信息。",
            },
        }
```

- 函数的操作是对runnable对象生成的结果进行进一步控制, 写了一个循环, 当其没有生成有效结果时, 会提示模型重新生成, 直到生成符合要求和规范的结果, 才终止循环, 返回结果, 这样的好处是保证了结果的健壮性, 降低因为偶发因素导致结果出错的概率

==与此前的确定性连边不同, 在更新节点之后, 需要进行条件路由(因为之后跳转到什么节点是不确定的, 可能是工具节点也可能是离开节点)==

#### 实际的工具调用节点(update_flight_safe_tools, update_flight_sensitive_tools): 对数据库增删改查

前面的更新节点已经得到了要调用什么工具, 则只需要经过路由判断, 即可来到对应的实际工具调用节点

- 实际调用工具函数时, 用create_tool_node_with_fallback(工具) 的方式, 目的是增加错误处理的相关操作

#### 离开节点(leave_skill)

同样是根据路由判断, 当CompleteOrEscalate工具中cancel的值为true, 就来到这个节点

- 其函数要实现的功能包括:
  - 提示模型正在返回主助理, 明确自己的身份并回顾之前的历史记录, 以更好地执行自己的任务
  - 对state的字段进行更新: state中的dialog_state指示了当前模型的身份, 此时由于是退出当前节点, 因此对其字段命名为"pop", 系统会自动识别出这是一个出栈命令而不是一个对字段的赋值

```python
return {
            # 更新对话状态为弹出 —— 这里的 pop不是具体值，而是表示要弹出（被识别为一个“命令”）,因此不冲突
            # 具体的值在state中定义，只能是那五个
            "dialog_state": "pop",
            "messages": messages,  # 返回消息列表
        }
```

#### 主助理节点(primary_assistant)

==**为什么要设置主助理:** 因为任务是多步骤的, 尽管一开始的fetch_user_info可以连接到各个节点, 但需要在各个助手间反复切换时, 就需要一个主助理协助调度==

其对应的函数中操作为: 设置提示词, 让模型判断应该将任务交给哪个子助手来完成

其可以使用的工具包括:

- 各个用于跳转到子助手的工具节点(信号型工具)
- 查询公司政策的工具
- 外部工具如tavily_tool
- 查询航班的工具(因为所有的信息都根据航班来进行, 所以主助理应该可以调用这个工具来查询需要的信息)

```
primary_assistant_tools = [
    tavily_tool,  # 假设TavilySearchResults是一个有效的搜索工具
    search_flights,  # 搜索航班的工具
    lookup_policy,  # 查找公司政策的工具
]
```

#### 各个节点之间的跳转关系

- 一开始的Start节点必然跳转到fetch_user_info
- fetch_user_info可能跳转到主助理与各个子助理, (如果state栈空则主助理)
- 主助理可能跳转到各个子助理的入口节点, 也可能跳转到主助理对应的工具节点
- 各个子助理的入口节点必然跳转到各个子助理节点
- 各个子助理节点可能跳转到工具节点或者leave节点
- leave节点必然跳转到主助理节点

### 执行部分

设置一个无限循环: 每次都由用户先输入问题, 再进入流程让模型进行处理

循环开始时, 判断用户的消息是不是表达"退出", 如果是, 直接结束流程

否则正式执行流程: 

```python
events = graph.stream({'messages': ('user', question)}, config, stream_mode='values')
```

使用stream方式, 这会返回发生中断之前的所有会话, 打印之

当中断发生后, 询问用户是否批准当前操作

- 如果批准, 直接继续执行(由于config中保存了会话id和用户id, 因此在中断后取得之前的config, 就可以回到原来的处理流程, 无需手动管理state, 非常方便)

```python
if user_input.strip().lower() == "y":
    # 之前的流中断了，因此要继续执行 —— 由于之前只是中断，因此会自动继续利用之前的state，继续流程
    events = graph.stream(None, config, stream_mode='values') // state的位置传None即可
```

- 如果不批准, 也要继续执行, 但多加一个tool消息(也是将局部作用域的提示伪装成工具调用返回给模型, 其中不仅提示模型请求被用户拒绝, 还将用户的输入作为拒绝的原因返回给模型)

## 项目亮点

### 将对子助理的身份提示放在tool_message中

实际上就是把人为对模型的提示"伪装成一条工具调用的返回结果", 从而给模型一个局部作用域的身份提示, 既避免了使用系统提示造成的污染(多个子助理各自修改系统提示, 由于其是全局的, 造成身份信息的混乱), 也避免了伪装成用户消息对用户消息造成的污染

### 使用闭包创建各个子助理的入口节点

四个子助理都要创建入口节点, 因此存在大量可复用的部分; 然而LangChain的绑定在node中的函数要求是必须仅以state作为输入, 因此==使用闭包, 达到"实际上传了多个参数"的效果==

### 将工具划分为安全工具与敏感工具

对于需要增删改查的操作, 认为是敏感操作, 流程图在执行到对应位置时会被人为中断, 向用户确认之后才继续执行

```python
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
```

### 使用循环 + 格式控制来确保得到符合规范的输出结果

模型是具有不稳定性的, 如果直接将单次LLM生成的结果返回, 其可能会由于各种意外因素导致不正确, 因此采取无限循环 + 格式判断的方式, 当且仅当模型返回了正确的格式, 才将结果返回. 

### 自定义信号型工具来提示子助理是否返回主助理

使用继承自BaseModel的类作为工具

### 用MemorySaver保存历史记录

每轮对话的记录会被自动添加到state的message字段中, 无需手动维护

此外, 在主程序中使用了 LangGraph 提供的 **持久化内存机制**: 

```python
memory = MemorySaver()
```

每一次状态更新后，它会把状态（如 `messages`）**自动保存到内存或磁盘**，支持==中断恢复、跨轮调用==.
