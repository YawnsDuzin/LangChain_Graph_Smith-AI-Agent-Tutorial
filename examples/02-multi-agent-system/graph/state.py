# graph/state.py
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ResearchData(TypedDict):
    """연구 데이터 구조"""
    topic: str
    sources: List[str]
    findings: str
    confidence: float


class DraftContent(TypedDict):
    """초안 콘텐츠 구조"""
    title: str
    sections: List[dict]
    word_count: int


class ReviewFeedback(TypedDict):
    """검토 피드백 구조"""
    score: float
    strengths: List[str]
    improvements: List[str]
    approved: bool


class AgentState(TypedDict):
    """멀티 에이전트 시스템 상태"""

    # 대화 메시지
    messages: Annotated[list, add_messages]

    # 작업 정보
    task: str
    task_type: str  # research, write, review, complete

    # 에이전트 결과
    research_data: Optional[ResearchData]
    draft_content: Optional[DraftContent]
    review_feedback: Optional[ReviewFeedback]

    # 워크플로우 제어
    next_agent: str
    iteration_count: int
    max_iterations: int

    # 최종 결과
    final_output: str
    status: str  # pending, in_progress, completed, failed


def create_initial_state(task: str) -> AgentState:
    """초기 상태 생성"""
    return AgentState(
        messages=[],
        task=task,
        task_type="pending",
        research_data=None,
        draft_content=None,
        review_feedback=None,
        next_agent="supervisor",
        iteration_count=0,
        max_iterations=5,
        final_output="",
        status="pending"
    )
