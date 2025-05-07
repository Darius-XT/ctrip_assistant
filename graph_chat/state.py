from typing import TypedDict, Annotated, Optional, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    更新对话状态栈。
    参数:
        left (list[str]): 当前的状态栈。
        right (Optional[str]): 想要添加到栈中的新状态或动作。如果为 None，则不做任何更改；
                               如果为 "pop"，则弹出栈顶元素；否则将该值添加到栈中。
    返回:
        list[str]: 更新后的状态栈。
    """
    if right is None:
        return left  # 如果right是None，保持当前状态栈不变
    if right == "pop":
        return left[:-1]  # 如果right是"pop"，移除栈顶元素（即最后一个状态）
    return left + [right]  # 否则，将right添加到状态栈中


# 状态类
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
