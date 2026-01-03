# graph/workflow.py
"""
멀티 에이전트 워크플로우 모듈

이 모듈은 LangGraph를 사용하여 Supervisor 패턴 기반의
멀티 에이전트 워크플로우를 구현합니다.

Supervisor 패턴 개념:
---------------------
Supervisor 패턴은 중앙 조율자(Supervisor)가 여러 전문 에이전트를 관리하는 구조입니다.

구조:
              ┌────────────────┐
      ┌──────>│   SUPERVISOR   │<──────┐
      │       └───────┬────────┘       │
      │               │                │
      │    ┌──────────┼──────────┐    │
      │    │          │          │    │
      │    ▼          ▼          ▼    │
  ┌────────────┐ ┌──────────┐ ┌────────────┐
  │ RESEARCHER │ │  WRITER  │ │  REVIEWER  │
  └─────┬──────┘ └────┬─────┘ └─────┬──────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
                      ▼
                  [Results]

워크플로우 흐름:
1. 사용자가 작업 요청
2. Supervisor가 작업 분석 후 적절한 에이전트 선택
3. 선택된 에이전트가 작업 수행
4. 결과를 Supervisor에게 반환
5. Supervisor가 다음 에이전트 선택 또는 완료 결정
6. 완료될 때까지 3-5 반복

이 패턴의 장점:
- 중앙 집중식 의사결정
- 명확한 책임 분리
- 동적 워크플로우 구성
- 쉬운 에이전트 추가/제거

LangGraph 개념 적용:
- StateGraph: 상태 기반 워크플로우 정의
- 조건부 엣지: Supervisor 결정에 따른 분기
- 순환 그래프: 에이전트 → Supervisor → 다른 에이전트
- 체크포인터: 대화 상태 유지
"""

from typing import Literal

# =============================================================================
# LangGraph 임포트 설명
# =============================================================================

# langgraph.graph.StateGraph
# ---------------------------
# 상태 기반 워크플로우 그래프를 정의하는 핵심 클래스입니다.
#
# 주요 메서드:
# - add_node(name, function): 노드 추가
#   - name: 노드 식별자
#   - function: state -> dict 형태의 함수
#
# - add_edge(from, to): 무조건 엣지 추가
#   - from: 시작 노드 (또는 START)
#   - to: 도착 노드 (또는 END)
#
# - add_conditional_edges(source, condition, mapping): 조건부 엣지
#   - source: 분기 시작 노드
#   - condition: state -> str 함수 (조건 판단)
#   - mapping: {조건값: 다음노드} 딕셔너리
#
# - compile(): 실행 가능한 앱으로 변환
#
# START, END:
# - START: 그래프 진입점
# - END: 그래프 종료점
from langgraph.graph import StateGraph, START, END

# langgraph.checkpoint.memory.MemorySaver
# ----------------------------------------
# 인메모리 체크포인터입니다.
#
# 체크포인터 역할:
# - 그래프 실행 상태 저장
# - thread_id별 독립적 상태 관리
# - 대화 기록 유지
#
# 프로덕션 대안:
# - PostgresSaver: PostgreSQL
# - SQLiteSaver: SQLite 파일
# - RedisSaver: Redis
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage
from graph.state import AgentState, create_initial_state
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.reviewer import ReviewerAgent
from agents.supervisor import SupervisorAgent

# langsmith.traceable
# --------------------
# LangSmith 추적 데코레이터입니다.
# 함수 실행을 LangSmith에 기록합니다.
from langsmith import traceable


class MultiAgentWorkflow:
    """
    멀티 에이전트 워크플로우 클래스

    Supervisor 패턴을 사용하여 여러 에이전트를 조율하는 워크플로우입니다.

    에이전트 구성:
    - Supervisor: 작업 분석 및 에이전트 선택
    - Researcher: 정보 수집 및 조사
    - Writer: 콘텐츠 작성
    - Reviewer: 품질 검토 및 피드백

    사용 예시:
        workflow = MultiAgentWorkflow()

        # 동기 실행
        result = workflow.run("AI에 대한 블로그 글 작성해줘")

        # 스트리밍 실행
        for event in workflow.stream("AI에 대한 블로그 글 작성해줘"):
            print(event)
    """

    def __init__(self):
        """
        워크플로우 초기화

        모든 에이전트를 초기화하고 그래프를 구축합니다.
        """
        # =====================================================================
        # 에이전트 초기화
        # =====================================================================
        # 각 에이전트는 특정 역할에 특화되어 있습니다.
        #
        # SupervisorAgent:
        # - 작업을 분석하고 다음 에이전트 결정
        # - 전체 워크플로우 진행 상황 모니터링
        # - 완료 여부 판단
        #
        # ResearcherAgent:
        # - 주제에 대한 정보 수집
        # - 검색 도구 사용
        # - 자료 정리 및 요약
        #
        # WriterAgent:
        # - 수집된 정보를 바탕으로 콘텐츠 작성
        # - 문서 구조화
        # - 초안 생성
        #
        # ReviewerAgent:
        # - 작성된 콘텐츠 검토
        # - 품질 평가
        # - 개선 피드백 제공
        # =====================================================================
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

        # 그래프 구축
        self.graph = self._build_graph()
        self.app = None  # compile() 후 설정됨

    def _build_graph(self) -> StateGraph:
        """
        그래프 구축

        노드와 엣지를 추가하여 워크플로우를 정의합니다.

        Returns:
            StateGraph: 구축된 그래프 (compile 전)

        그래프 구조:
        1. START → supervisor
        2. supervisor → (조건부) → researcher/writer/reviewer/END
        3. researcher/writer/reviewer → supervisor (복귀)

        순환 그래프:
        - 에이전트들이 supervisor를 통해 순환
        - supervisor가 "finish" 결정할 때까지 반복
        """
        # =====================================================================
        # StateGraph 생성
        # =====================================================================
        # StateGraph(StateType)
        #
        # AgentState는 TypedDict로 정의된 상태 스키마입니다.
        # 모든 노드가 이 상태를 공유하고 업데이트합니다.
        #
        # AgentState 구조 (graph/state.py 참조):
        # - messages: 대화 기록
        # - task: 원본 작업
        # - next_agent: 다음에 실행할 에이전트
        # - research_data: 수집된 정보
        # - draft_content: 작성된 초안
        # - review_feedback: 검토 피드백
        # - iteration_count: 반복 횟수
        # =====================================================================
        graph = StateGraph(AgentState)

        # =====================================================================
        # 노드 추가 (add_node)
        # =====================================================================
        # graph.add_node(name, function)
        #
        # 노드 함수 시그니처:
        #   def node_fn(state: AgentState) -> dict
        #
        # 반환 규칙:
        # - 업데이트할 필드만 포함하는 dict 반환
        # - 반환하지 않은 필드는 그대로 유지
        #
        # 예시:
        #   def my_node(state):
        #       return {"field1": new_value}  # field1만 업데이트
        # =====================================================================
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("reviewer", self._reviewer_node)

        # =====================================================================
        # 시작 엣지 추가
        # =====================================================================
        # graph.add_edge(START, "supervisor")
        #
        # START는 특수 노드로, 그래프의 진입점을 나타냅니다.
        # 이 엣지는 그래프가 supervisor 노드에서 시작함을 의미합니다.
        # =====================================================================
        graph.add_edge(START, "supervisor")

        # =====================================================================
        # 조건부 엣지 추가 (Supervisor 분기)
        # =====================================================================
        # graph.add_conditional_edges(
        #     source_node,    # 분기 시작 노드
        #     condition_fn,   # 조건 판단 함수
        #     mapping         # 조건값 → 다음 노드 매핑
        # )
        #
        # 조건 함수 (condition_fn):
        # - 입력: state (현재 상태)
        # - 출력: 문자열 (mapping의 키)
        # - 상태를 분석하여 다음 노드 결정
        #
        # 매핑 (mapping):
        # - 조건 함수 반환값 → 다음 노드 이름
        # - "finish" → END는 그래프 종료를 의미
        #
        # Supervisor 패턴에서의 역할:
        # - Supervisor가 상태의 "next_agent" 필드 설정
        # - _route_from_supervisor가 이 값을 읽어 분기
        # - 동적으로 다음 에이전트 결정
        # =====================================================================
        graph.add_conditional_edges(
            "supervisor",                    # 분기 시작점
            self._route_from_supervisor,     # 조건 판단 함수
            {
                "researcher": "researcher",  # 연구 에이전트로
                "writer": "writer",          # 작성 에이전트로
                "reviewer": "reviewer",      # 검토 에이전트로
                "finish": END               # 워크플로우 종료
            }
        )

        # =====================================================================
        # 에이전트 → Supervisor 복귀 엣지
        # =====================================================================
        # 각 에이전트가 작업 완료 후 supervisor로 돌아갑니다.
        # 이를 통해 순환 구조가 형성됩니다.
        #
        # 순환 흐름 예시:
        # 1. supervisor → researcher (조사 필요)
        # 2. researcher → supervisor (조사 완료)
        # 3. supervisor → writer (작성 필요)
        # 4. writer → supervisor (작성 완료)
        # 5. supervisor → reviewer (검토 필요)
        # 6. reviewer → supervisor (검토 완료)
        # 7. supervisor → END (완료)
        # =====================================================================
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("reviewer", "supervisor")

        return graph

    # =========================================================================
    # 노드 함수들
    # =========================================================================
    # 각 노드 함수는 state를 입력받아 에이전트를 실행하고
    # 상태 업데이트를 반환합니다.
    # =========================================================================

    def _supervisor_node(self, state: AgentState) -> dict:
        """
        슈퍼바이저 노드

        현재 상태를 분석하고 다음 에이전트를 결정합니다.

        Args:
            state (AgentState): 현재 상태

        Returns:
            dict: 상태 업데이트 (next_agent 포함)
        """
        return self.supervisor.invoke(state)

    def _researcher_node(self, state: AgentState) -> dict:
        """
        연구 에이전트 노드

        주제에 대한 정보를 수집합니다.

        Args:
            state (AgentState): 현재 상태

        Returns:
            dict: 상태 업데이트 (research_data 포함)
        """
        return self.researcher.invoke(state)

    def _writer_node(self, state: AgentState) -> dict:
        """
        작성 에이전트 노드

        수집된 정보를 바탕으로 콘텐츠를 작성합니다.

        Args:
            state (AgentState): 현재 상태

        Returns:
            dict: 상태 업데이트 (draft_content 포함)
        """
        return self.writer.invoke(state)

    def _reviewer_node(self, state: AgentState) -> dict:
        """
        검토 에이전트 노드

        작성된 콘텐츠를 검토하고 피드백을 제공합니다.

        Args:
            state (AgentState): 현재 상태

        Returns:
            dict: 상태 업데이트 (review_feedback 포함)
        """
        return self.reviewer.invoke(state)

    def _route_from_supervisor(
        self,
        state: AgentState
    ) -> Literal["researcher", "writer", "reviewer", "finish"]:
        """
        슈퍼바이저 라우팅 함수

        supervisor 노드 실행 후 다음 노드를 결정합니다.

        Args:
            state (AgentState): 현재 상태

        Returns:
            Literal["researcher", "writer", "reviewer", "finish"]:
                다음 노드 키

        조건부 라우팅 상세:
        -------------------
        1. supervisor 노드가 상태의 "next_agent" 필드를 설정
        2. 이 함수가 해당 값을 읽어 반환
        3. add_conditional_edges의 mapping에서 다음 노드 결정

        Literal 타입:
        - 반환값을 특정 문자열로 제한
        - 타입 체킹과 자동완성 지원
        - mapping의 키와 일치해야 함
        """
        # 상태에서 다음 에이전트 읽기
        next_agent = state.get("next_agent", "").lower()

        # 에이전트 이름에 따라 라우팅
        if next_agent == "researcher":
            return "researcher"
        elif next_agent == "writer":
            return "writer"
        elif next_agent == "reviewer":
            return "reviewer"
        else:
            # 인식되지 않는 값이면 종료
            return "finish"

    def compile(self, with_memory: bool = True):
        """
        그래프 컴파일

        구축된 그래프를 실행 가능한 앱으로 변환합니다.

        Args:
            with_memory (bool): 체크포인터 사용 여부

        Returns:
            CompiledGraph: 실행 가능한 그래프

        compile() 상세:
        ---------------
        graph.compile(checkpointer=None)

        checkpointer 효과:
        - 각 노드 실행 후 상태 저장
        - thread_id별 독립적 상태 관리
        - 중단 후 재개 가능

        컴파일된 앱의 메서드:
        - invoke(input, config): 동기 실행
        - ainvoke(input, config): 비동기 실행
        - stream(input, config): 스트리밍 실행
        - astream(input, config): 비동기 스트리밍
        """
        if with_memory:
            checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=checkpointer)
        else:
            self.app = self.graph.compile()
        return self.app

    @traceable(name="multi_agent_run")
    def run(self, task: str, thread_id: str = "default") -> dict:
        """
        워크플로우 실행

        주어진 작업을 멀티 에이전트가 협력하여 수행합니다.

        Args:
            task (str): 수행할 작업 설명
            thread_id (str): 대화 스레드 ID (체크포인터용)

        Returns:
            dict: 실행 결과
                - final_output: 최종 출력
                - research_data: 수집된 정보
                - draft_content: 작성된 콘텐츠
                - review_feedback: 검토 피드백
                - iterations: 반복 횟수
                - status: 완료 상태

        @traceable 데코레이터:
        ----------------------
        @traceable(name="multi_agent_run")

        이 메서드 호출이 LangSmith에 "multi_agent_run" 이름으로 기록됩니다.
        - 전체 워크플로우 추적
        - 각 에이전트 호출 중첩 표시
        - 성능 분석 용이
        """
        # 앱이 컴파일되지 않았으면 컴파일
        if self.app is None:
            self.compile()

        # 초기 상태 생성
        initial_state = create_initial_state(task)
        initial_state["messages"] = [HumanMessage(content=task)]

        # =====================================================================
        # app.invoke() 상세 설명
        # =====================================================================
        # invoke(input, config)
        #
        # input: 초기 상태 딕셔너리
        # config: 실행 설정
        #   - configurable: 런타임 설정
        #     - thread_id: 대화 스레드 식별자
        #
        # 실행 흐름:
        # 1. START 노드에서 시작
        # 2. 각 노드 순차 실행
        # 3. 조건부 엣지에서 분기
        # 4. END 도달 시 종료
        # 5. 최종 상태 반환
        #
        # 체크포인터가 있으면:
        # - thread_id로 이전 상태 로드 가능
        # - 각 노드 실행 후 상태 저장
        # =====================================================================
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke(initial_state, config=config)

        # 결과 정리
        return {
            "final_output": result.get("draft_content", {}).get("title", ""),
            "research_data": result.get("research_data"),
            "draft_content": result.get("draft_content"),
            "review_feedback": result.get("review_feedback"),
            "iterations": result.get("iteration_count", 0),
            "status": "completed" if result.get("review_feedback", {}).get("approved") else "needs_revision"
        }

    def stream(self, task: str, thread_id: str = "default"):
        """
        스트리밍 실행

        워크플로우를 실행하면서 각 노드의 결과를 실시간으로 반환합니다.

        Args:
            task (str): 수행할 작업
            thread_id (str): 스레드 ID

        Yields:
            dict: 각 노드의 실행 결과

        stream() 상세 설명:
        --------------------
        app.stream(input, config, stream_mode="updates")

        Generator로 각 노드 결과를 순차 반환합니다.

        stream_mode 옵션:
        - "values": 각 단계의 전체 상태
        - "updates": 각 노드가 업데이트한 부분만
        - "debug": 디버그 정보 포함

        스트리밍 활용:
        - 실시간 UI 업데이트
        - 진행 상황 표시
        - 중간 결과 활용
        """
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(task)
        initial_state["messages"] = [HumanMessage(content=task)]

        config = {"configurable": {"thread_id": thread_id}}

        # stream: Generator로 각 노드 결과 반환
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


# =============================================================================
# 사용 예시 및 테스트
# =============================================================================
if __name__ == "__main__":
    """
    워크플로우 테스트

    실행 방법:
        python -m graph.workflow
    """
    workflow = MultiAgentWorkflow()
    print(workflow.visualize())

    # 실행
    print("\n=== 워크플로우 실행 ===")
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
