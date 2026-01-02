# assistant/graph/state.py
from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ToolCall(TypedDict):
    """도구 호출 정보"""
    tool_name: str
    tool_args: dict
    requires_approval: bool
    approved: Optional[bool]


class AssistantState(TypedDict):
    """어시스턴트 상태"""
    # 대화 메시지
    messages: Annotated[list, add_messages]

    # 현재 사용자 입력
    user_input: str

    # 의도 분석 결과
    intent: str  # chat, tool_use, clarify

    # 도구 관련
    pending_tool_calls: List[ToolCall]
    tool_results: List[dict]

    # 워크플로우 제어
    requires_approval: bool
    user_approved: Optional[bool]

    # 응답
    response: str
    streaming: bool

    # 세션 정보
    session_id: str


def create_initial_state(
    user_input: str,
    session_id: str = "default"
) -> AssistantState:
    """초기 상태 생성"""
    return AssistantState(
        messages=[],
        user_input=user_input,
        intent="",
        pending_tool_calls=[],
        tool_results=[],
        requires_approval=False,
        user_approved=None,
        response="",
        streaming=True,
        session_id=session_id
    )
