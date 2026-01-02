# graph/workflow.py
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from graph.state import AgentState, create_initial_state
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.reviewer import ReviewerAgent
from agents.supervisor import SupervisorAgent
from langsmith import traceable


class MultiAgentWorkflow:
    """멀티 에이전트 워크플로우"""

    def __init__(self):
        # 에이전트 초기화
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

        # 그래프 구축
        self.graph = self._build_graph()
        self.app = None

    def _build_graph(self) -> StateGraph:
        """그래프 구축"""
        graph = StateGraph(AgentState)

        # 노드 추가
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("reviewer", self._reviewer_node)

        # 시작점
        graph.add_edge(START, "supervisor")

        # 슈퍼바이저에서 조건부 라우팅
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "researcher": "researcher",
                "writer": "writer",
                "reviewer": "reviewer",
                "finish": END
            }
        )

        # 각 에이전트 후 슈퍼바이저로 복귀
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("reviewer", "supervisor")

        return graph

    def _supervisor_node(self, state: AgentState) -> dict:
        """슈퍼바이저 노드"""
        return self.supervisor.invoke(state)

    def _researcher_node(self, state: AgentState) -> dict:
        """연구 에이전트 노드"""
        return self.researcher.invoke(state)

    def _writer_node(self, state: AgentState) -> dict:
        """작성 에이전트 노드"""
        return self.writer.invoke(state)

    def _reviewer_node(self, state: AgentState) -> dict:
        """검토 에이전트 노드"""
        return self.reviewer.invoke(state)

    def _route_from_supervisor(
        self,
        state: AgentState
    ) -> Literal["researcher", "writer", "reviewer", "finish"]:
        """슈퍼바이저 라우팅"""
        next_agent = state.get("next_agent", "").lower()

        if next_agent == "researcher":
            return "researcher"
        elif next_agent == "writer":
            return "writer"
        elif next_agent == "reviewer":
            return "reviewer"
        else:
            return "finish"

    def compile(self, with_memory: bool = True):
        """그래프 컴파일"""
        if with_memory:
            checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=checkpointer)
        else:
            self.app = self.graph.compile()
        return self.app

    @traceable(name="multi_agent_run")
    def run(self, task: str, thread_id: str = "default") -> dict:
        """워크플로우 실행"""
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(task)
        initial_state["messages"] = [HumanMessage(content=task)]

        config = {"configurable": {"thread_id": thread_id}}

        result = self.app.invoke(initial_state, config=config)

        return {
            "final_output": result.get("draft_content", {}).get("title", ""),
            "research_data": result.get("research_data"),
            "draft_content": result.get("draft_content"),
            "review_feedback": result.get("review_feedback"),
            "iterations": result.get("iteration_count", 0),
            "status": "completed" if result.get("review_feedback", {}).get("approved") else "needs_revision"
        }

    def stream(self, task: str, thread_id: str = "default"):
        """스트리밍 실행"""
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(task)
        initial_state["messages"] = [HumanMessage(content=task)]

        config = {"configurable": {"thread_id": thread_id}}

        for event in self.app.stream(initial_state, config=config, stream_mode="updates"):
            yield event

    def visualize(self) -> str:
        """그래프 시각화 (ASCII)"""
        return """
    ┌─────────────────────────────────────────────────────────┐
    │                  멀티 에이전트 워크플로우                  │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │                        ┌─────────┐                      │
    │                        │  START  │                      │
    │                        └────┬────┘                      │
    │                             │                           │
    │                             ▼                           │
    │                    ┌────────────────┐                   │
    │         ┌──────────│  SUPERVISOR    │◄─────────┐       │
    │         │          └───────┬────────┘          │       │
    │         │                  │                   │       │
    │         │     ┌────────────┼────────────┐     │       │
    │         │     │            │            │     │       │
    │         ▼     ▼            ▼            ▼     │       │
    │     ┌────────────┐   ┌──────────┐   ┌────────────┐   │
    │     │ RESEARCHER │   │  WRITER  │   │  REVIEWER  │   │
    │     └─────┬──────┘   └────┬─────┘   └─────┬──────┘   │
    │           │               │               │          │
    │           └───────────────┴───────────────┘          │
    │                           │                           │
    │                    finish │                           │
    │                           ▼                           │
    │                      ┌────────┐                       │
    │                      │  END   │                       │
    │                      └────────┘                       │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
        """


# 사용 예시
if __name__ == "__main__":
    workflow = MultiAgentWorkflow()
    print(workflow.visualize())

    # 실행
    result = workflow.run("인공지능의 미래에 대한 블로그 글을 작성해주세요.")

    print("\n=== 실행 결과 ===")
    print(f"상태: {result['status']}")
    print(f"반복 횟수: {result['iterations']}")

    if result['draft_content']:
        print(f"\n제목: {result['draft_content'].get('title', 'N/A')}")
        print(f"단어 수: {result['draft_content'].get('word_count', 0)}")

    if result['review_feedback']:
        print(f"\n검토 점수: {result['review_feedback'].get('score', 0)}")
        print(f"승인 여부: {result['review_feedback'].get('approved', False)}")
