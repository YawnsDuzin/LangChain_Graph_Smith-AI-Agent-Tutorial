# assistant/graph/assistant.py
"""
============================================================================
AI 어시스턴트 메인 모듈 - LangGraph 기반 대화형 에이전트
============================================================================

이 모듈은 LangGraph를 사용하여 도구 사용이 가능한 대화형 AI 어시스턴트를 구현합니다.

핵심 개념:
    - StateGraph: 상태 기반의 그래프 워크플로우
    - 노드(Node): 작업을 수행하는 함수
    - 엣지(Edge): 노드 간의 연결과 실행 흐름
    - 조건부 엣지: 상태에 따라 다음 노드를 결정
    - 체크포인팅: 대화 상태 저장 및 복원

워크플로우 구조:
    START → process_input → assistant → [분기]
                                         ├→ tools → assistant (도구 사용)
                                         ├→ check_approval → tools/respond (승인 필요)
                                         └→ generate_response → END (응답)

사용 예시:
    from assistant.graph.assistant import AIAssistant

    assistant = AIAssistant()
    response = assistant.chat("오늘 날씨 어때?", session_id="user-123")
    print(response)

============================================================================
"""
from typing import Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langsmith import traceable

from assistant.graph.state import AssistantState, create_initial_state
from assistant.tools import ALL_TOOLS, TOOLS_REQUIRING_APPROVAL
from assistant.memory.conversation import conversation_store


# ============================================================================
# AI 어시스턴트 클래스
# ============================================================================
class AIAssistant:
    """
    ========================================================================
    대화형 AI 어시스턴트
    ========================================================================

    LangGraph를 사용하여 도구 사용, 승인 프로세스, 스트리밍을 지원하는
    대화형 AI 어시스턴트를 구현합니다.

    주요 기능:
        - 웹 검색: 최신 정보 검색
        - 계산: 수학 계산 및 단위 변환
        - 코드 실행: Python 코드 실행 (승인 필요)
        - 대화 기록: 세션별 대화 저장

    아키텍처:
        1. 입력 처리: 사용자 입력을 메시지로 변환
        2. LLM 호출: 응답 또는 도구 호출 결정
        3. 도구 실행: 필요 시 도구 실행 및 결과 수집
        4. 응답 생성: 최종 응답 생성 및 저장

    사용 예시:
        # 기본 사용
        assistant = AIAssistant()
        response = assistant.chat("안녕하세요!")

        # 스트리밍
        for chunk in assistant.stream_chat("오늘 뉴스 알려줘"):
            print(chunk)

        # 비동기
        response = await assistant.achat("날씨 어때?")
    ========================================================================
    """

    # ========================================================================
    # 시스템 프롬프트
    # ========================================================================
    # 시스템 프롬프트는 AI의 성격, 능력, 규칙을 정의합니다.
    # 모든 대화에서 이 프롬프트가 가장 먼저 전달됩니다.
    #
    # 좋은 시스템 프롬프트의 요소:
    #   1. 역할 정의: "당신은 ...입니다"
    #   2. 능력 명시: 할 수 있는 것들 나열
    #   3. 규칙/제약: 따라야 할 지침
    #   4. 출력 형식: 응답 스타일 지정
    # ========================================================================
    SYSTEM_PROMPT = """당신은 유능하고 친절한 AI 어시스턴트입니다.

당신의 능력:
1. 🔍 인터넷 검색으로 최신 정보 제공
2. 🧮 수학 계산 및 단위 변환
3. 🐍 Python 코드 실행 및 설명
4. 📅 날짜/시간 계산
5. 📝 텍스트 요약

규칙:
- 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.
- 필요한 경우 도구를 사용하여 정보를 얻으세요.
- 모르는 것은 솔직히 모른다고 말하세요.
- 친근하지만 전문적인 톤을 유지하세요.
- 한국어로 응답하세요.

중요: 코드 실행이 필요한 경우, 사용자에게 먼저 확인을 받으세요."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        """
        ====================================================================
        AIAssistant 초기화
        ====================================================================

        Args:
            model: 사용할 OpenAI 모델 (기본값: "gpt-4")
            temperature: 응답의 창의성 (0.0~2.0, 기본값: 0.7)

        초기화 과정:
            1. 모델 설정 저장
            2. 도구 목록 설정
            3. ChatOpenAI 인스턴스 생성 및 도구 바인딩
            4. 그래프 구축
        ====================================================================
        """
        self.model = model
        self.temperature = temperature
        self.tools = ALL_TOOLS

        # ====================================================================
        # ChatOpenAI 인스턴스 생성
        # ====================================================================
        # ChatOpenAI 상세 설명:
        #
        # langchain_openai 패키지의 핵심 클래스로, OpenAI의 채팅 모델을 래핑합니다.
        #
        # 주요 매개변수:
        #   - model (str): 사용할 모델 ID
        #       예: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
        #
        #   - temperature (float): 응답의 무작위성 (0.0~2.0)
        #       0.0: 결정론적, 항상 같은 응답
        #       1.0: 적당한 다양성
        #       2.0: 매우 무작위
        #
        #   - streaming (bool): 스트리밍 모드 활성화
        #       True: 응답을 토큰 단위로 스트리밍
        #       False: 전체 응답을 한 번에 반환
        #
        #   - max_tokens (int, optional): 최대 출력 토큰 수
        #
        #   - api_key (str, optional): OpenAI API 키
        #       설정 안하면 OPENAI_API_KEY 환경 변수 사용
        #
        # 기타 매개변수:
        #   - model_kwargs: 추가 모델 파라미터 (dict)
        #   - request_timeout: 요청 타임아웃 (초)
        #   - max_retries: 재시도 횟수
        # ====================================================================
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True  # 스트리밍 활성화
        ).bind_tools(self.tools)
        # ====================================================================
        # bind_tools() 상세 설명:
        #
        # LLM에 사용 가능한 도구(함수)를 연결하는 메서드입니다.
        # 이를 통해 LLM이 필요할 때 도구를 호출할 수 있습니다.
        #
        # 동작 원리:
        #   1. 도구 정의를 OpenAI의 function calling 형식으로 변환
        #   2. LLM 호출 시 functions 파라미터에 포함
        #   3. LLM이 도구 호출이 필요하다고 판단하면 tool_calls 반환
        #
        # 도구 정의 형식:
        #   @tool 데코레이터로 정의된 함수는 자동으로 변환됩니다.
        #
        #   @tool
        #   def search(query: str) -> str:
        #       """인터넷 검색"""
        #       ...
        #
        #   변환 결과:
        #   {
        #       "name": "search",
        #       "description": "인터넷 검색",
        #       "parameters": {
        #           "type": "object",
        #           "properties": {
        #               "query": {"type": "string"}
        #           },
        #           "required": ["query"]
        #       }
        #   }
        #
        # 사용 예시:
        #   llm_with_tools = llm.bind_tools([search_tool, calc_tool])
        #   response = llm_with_tools.invoke(messages)
        #
        #   if response.tool_calls:
        #       for call in response.tool_calls:
        #           tool_name = call["name"]
        #           tool_args = call["args"]
        # ====================================================================

        # 그래프 구축
        self.graph = self._build_graph()
        self.app = None  # compile() 호출 시 설정됨

    def _build_graph(self) -> StateGraph:
        """
        ====================================================================
        LangGraph 그래프 구축
        ====================================================================

        StateGraph 상세 설명:
        --------------------------------------------------------------------
        StateGraph는 LangGraph의 핵심 클래스로, 상태 기반 워크플로우를 정의합니다.

        생성:
            graph = StateGraph(StateType)
            - StateType: 상태의 타입 (TypedDict 클래스)

        노드 추가 (add_node):
            graph.add_node("노드이름", 함수)
            - 노드 이름: 그래프 내에서 고유한 식별자
            - 함수: (state) -> dict 형태
                - state: 현재 상태
                - 반환값: 상태 업데이트 (일부 필드만 포함 가능)

        엣지 추가 (add_edge):
            graph.add_edge("소스노드", "타겟노드")
            - 소스 노드 실행 후 항상 타겟 노드 실행

        조건부 엣지 (add_conditional_edges):
            graph.add_conditional_edges(
                "소스노드",
                라우팅_함수,
                {"조건1": "노드1", "조건2": "노드2"}
            )
            - 라우팅_함수: (state) -> str (조건 반환)
            - 조건에 따라 다음 노드 결정

        특수 노드:
            - START: 그래프 시작점
            - END: 그래프 종료점

        Returns:
            StateGraph: 구축된 그래프 (아직 컴파일되지 않음)
        ====================================================================
        """
        # StateGraph 생성
        # AssistantState 타입으로 상태 스키마 정의
        graph = StateGraph(AssistantState)

        # ====================================================================
        # 노드 추가
        # ====================================================================
        # add_node(이름, 함수) 상세 설명:
        #
        # 그래프에 노드를 추가합니다. 각 노드는 상태를 받아 처리하는 함수입니다.
        #
        # 노드 함수 규칙:
        #   1. 입력: state (TypedDict 인스턴스)
        #   2. 출력: dict (상태 업데이트)
        #   3. 반환된 키만 상태에 병합됨
        #
        # 예시:
        #   def my_node(state):
        #       return {"messages": [new_msg]}  # messages만 업데이트
        # ====================================================================
        graph.add_node("process_input", self._process_input)
        graph.add_node("assistant", self._assistant_node)

        # ToolNode - LangGraph에서 제공하는 도구 실행 노드
        # ====================================================================
        # ToolNode 상세 설명:
        #
        # LangGraph의 prebuilt 모듈에서 제공하는 도구 실행 노드입니다.
        # tool_calls가 있는 AIMessage를 처리하여 도구를 실행합니다.
        #
        # 동작 과정:
        #   1. 상태에서 마지막 메시지의 tool_calls 확인
        #   2. 각 tool_call에 대해 해당 도구 함수 실행
        #   3. ToolMessage로 결과 래핑
        #   4. messages 리스트에 추가
        #
        # 생성:
        #   ToolNode(tools)
        #   - tools: @tool 데코레이터로 정의된 함수 리스트
        #
        # 자동 처리:
        #   - 도구 이름으로 함수 매칭
        #   - 인자 전달 및 실행
        #   - 예외 처리 및 에러 메시지 생성
        #   - ToolMessage 생성 (tool_call_id 포함)
        # ====================================================================
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("check_approval", self._check_approval)
        graph.add_node("generate_response", self._generate_response)

        # ====================================================================
        # 엣지 추가
        # ====================================================================
        # add_edge(소스, 타겟):
        #   소스 노드 실행 완료 후 항상 타겟 노드로 이동
        #
        # START와 END:
        #   - START: 그래프 진입점. 첫 번째 노드로의 엣지 시작점
        #   - END: 그래프 종료점. 이 노드에 도달하면 실행 종료
        # ====================================================================
        graph.add_edge(START, "process_input")
        graph.add_edge("process_input", "assistant")

        # ====================================================================
        # 조건부 엣지 - 어시스턴트 후 라우팅
        # ====================================================================
        # add_conditional_edges 상세 설명:
        #
        # 상태에 따라 다음 노드를 동적으로 결정합니다.
        #
        # 매개변수:
        #   1. source (str): 시작 노드 이름
        #   2. path (Callable): 라우팅 함수
        #       - 입력: state
        #       - 출력: 다음 노드 이름 (str)
        #   3. path_map (dict, optional): 라우팅 결과를 노드 이름으로 매핑
        #       - 라우팅 함수 반환값 → 실제 노드 이름
        #
        # 라우팅 함수 규칙:
        #   - Literal 타입으로 가능한 반환값 명시 권장
        #   - 반환값은 path_map의 키와 일치해야 함
        #
        # 예시:
        #   def router(state) -> Literal["a", "b"]:
        #       if state["condition"]:
        #           return "a"
        #       return "b"
        #
        #   graph.add_conditional_edges(
        #       "node1",
        #       router,
        #       {"a": "node_a", "b": "node_b"}
        #   )
        # ====================================================================
        graph.add_conditional_edges(
            "assistant",
            self._route_after_assistant,
            {
                "tools": "tools",           # 일반 도구 → 바로 실행
                "check_approval": "check_approval",  # 승인 필요 도구 → 승인 확인
                "respond": "generate_response"  # 도구 불필요 → 응답 생성
            }
        )

        # 도구 실행 후 어시스턴트로 복귀 (결과를 보고 추가 도구 호출 또는 응답)
        graph.add_edge("tools", "assistant")

        # 승인 확인 후 라우팅
        graph.add_conditional_edges(
            "check_approval",
            self._route_after_approval,
            {
                "tools": "tools",           # 승인됨 → 도구 실행
                "respond": "generate_response"  # 거부됨 → 응답 생성
            }
        )

        # 응답 생성 후 종료
        graph.add_edge("generate_response", END)

        return graph

    # ========================================================================
    # 노드 함수들
    # ========================================================================

    @traceable(name="process_input")
    def _process_input(self, state: AssistantState) -> dict:
        """
        ====================================================================
        입력 처리 노드
        ====================================================================

        @traceable 데코레이터:
        --------------------------------------------------------------------
        LangSmith에서 제공하는 데코레이터로, 함수 실행을 자동으로 추적합니다.

        기능:
            - 함수 호출 시작/종료 시간 기록
            - 입력 인자 및 반환값 저장
            - 예외 발생 시 스택 트레이스 저장
            - LangSmith 대시보드에서 조회 가능

        매개변수:
            - name (str): 트레이스에 표시될 이름
            - run_type (str): 실행 유형 (chain, llm, tool, retriever 등)
            - metadata (dict): 추가 메타데이터

        사용 예시:
            @traceable(name="my_function", metadata={"version": "1.0"})
            def my_function(x):
                return x * 2

        LangSmith에서 확인:
            1. smith.langchain.com 접속
            2. 프로젝트 선택
            3. Runs 탭에서 트레이스 확인

        Args:
            state: 현재 그래프 상태

        Returns:
            dict: 상태 업데이트 (messages에 HumanMessage 추가)
        ====================================================================
        """
        user_input = state["user_input"]
        session_id = state["session_id"]

        # 대화 기록 가져오기 및 사용자 메시지 추가
        memory = conversation_store.get_session(session_id)
        memory.add_user_message(user_input)

        # HumanMessage로 변환하여 messages에 추가
        # ====================================================================
        # HumanMessage 상세 설명:
        #
        # langchain_core.messages의 메시지 타입 중 하나입니다.
        # 사용자의 입력을 나타냅니다.
        #
        # 메시지 타입들:
        #   - SystemMessage: 시스템 지시 (AI 역할 정의 등)
        #   - HumanMessage: 사용자 입력
        #   - AIMessage: AI 응답
        #   - ToolMessage: 도구 실행 결과
        #   - FunctionMessage: (deprecated) ToolMessage 사용
        #
        # 생성:
        #   msg = HumanMessage(content="안녕하세요!")
        #
        # 속성:
        #   - content (str | list): 메시지 내용
        #   - additional_kwargs (dict): 추가 데이터
        #   - id (str, optional): 고유 식별자
        # ====================================================================
        return {
            "messages": [HumanMessage(content=user_input)]
        }

    @traceable(name="assistant")
    def _assistant_node(self, state: AssistantState) -> dict:
        """
        ====================================================================
        어시스턴트 노드 - LLM 호출
        ====================================================================

        LLM을 호출하여 응답 또는 도구 호출을 생성합니다.

        처리 과정:
            1. 시스템 프롬프트 + 대화 기록으로 메시지 구성
            2. LLM에 메시지 전달
            3. 응답 받아서 messages에 추가

        LLM 응답 종류:
            1. 일반 응답: content에 텍스트
            2. 도구 호출: tool_calls에 호출 정보

        Args:
            state: 현재 그래프 상태

        Returns:
            dict: 상태 업데이트 (messages에 AIMessage 추가)
        ====================================================================
        """
        # 시스템 프롬프트 + 대화 기록
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]

        # LLM 호출
        # ====================================================================
        # invoke() 상세 설명:
        #
        # LLM에 메시지를 전달하고 응답을 받는 동기 메서드입니다.
        #
        # 입력:
        #   - messages: 메시지 리스트 또는 단일 문자열
        #
        # 출력:
        #   - AIMessage 인스턴스
        #       - content: 응답 텍스트
        #       - tool_calls: 도구 호출 정보 (있을 경우)
        #       - additional_kwargs: 추가 정보
        #
        # 도구 호출 응답 예시:
        #   AIMessage(
        #       content="",
        #       tool_calls=[
        #           {
        #               "id": "call_xxx",
        #               "name": "search",
        #               "args": {"query": "날씨"}
        #           }
        #       ]
        #   )
        # ====================================================================
        response = self.llm.invoke(messages)

        return {"messages": [response]}

    def _route_after_assistant(
        self,
        state: AssistantState
    ) -> Literal["tools", "check_approval", "respond"]:
        """
        ====================================================================
        어시스턴트 후 라우팅 결정
        ====================================================================

        Literal 타입 힌트:
        --------------------------------------------------------------------
        Literal은 함수가 반환할 수 있는 값을 명시적으로 지정합니다.

        장점:
            - 타입 검사기가 잘못된 반환값 감지
            - IDE에서 가능한 값 자동완성
            - 코드 문서화 효과

        사용 예시:
            def get_status() -> Literal["pending", "done", "error"]:
                ...

        라우팅 로직:
            1. 마지막 메시지에 tool_calls가 있는지 확인
            2. tool_calls가 있으면:
               - 승인 필요 도구가 있으면 → "check_approval"
               - 없으면 → "tools"
            3. tool_calls가 없으면 → "respond"

        Args:
            state: 현재 그래프 상태

        Returns:
            다음 노드 이름
        ====================================================================
        """
        last_message = state["messages"][-1]

        # 도구 호출이 있는지 확인
        # ====================================================================
        # tool_calls 확인:
        #
        # AIMessage에 tool_calls 속성이 있으면 LLM이 도구 사용을 요청한 것입니다.
        #
        # tool_calls 구조:
        #   [
        #       {
        #           "id": "call_abc123",  # 고유 ID
        #           "name": "search",      # 도구 이름
        #           "args": {"query": "..."} # 인자
        #       },
        #       ...
        #   ]
        #
        # hasattr로 확인하는 이유:
        #   - 메시지 타입이 AIMessage가 아닐 수 있음
        #   - 이전 버전 호환성
        # ====================================================================
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls

            # 승인이 필요한 도구가 있는지 확인
            for call in tool_calls:
                if call["name"] in TOOLS_REQUIRING_APPROVAL:
                    return "check_approval"

            return "tools"

        return "respond"

    @traceable(name="check_approval")
    def _check_approval(self, state: AssistantState) -> dict:
        """
        ====================================================================
        승인 확인 노드
        ====================================================================

        위험한 도구 실행 전 사용자 승인을 요청합니다.

        처리 과정:
            1. 대기 중인 도구 호출 정보 수집
            2. 각 도구에 대해 승인 필요 여부 표시
            3. requires_approval 플래그 설정

        Args:
            state: 현재 그래프 상태

        Returns:
            dict: 상태 업데이트 (pending_tool_calls, requires_approval)
        ====================================================================
        """
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        pending_calls = []
        for call in tool_calls:
            pending_calls.append({
                "tool_name": call["name"],
                "tool_args": call["args"],
                "requires_approval": call["name"] in TOOLS_REQUIRING_APPROVAL,
                "approved": None
            })

        # 승인 대기 상태로 설정
        return {
            "pending_tool_calls": pending_calls,
            "requires_approval": True
        }

    def _route_after_approval(
        self,
        state: AssistantState
    ) -> Literal["tools", "respond"]:
        """
        ====================================================================
        승인 후 라우팅 결정
        ====================================================================

        Args:
            state: 현재 그래프 상태

        Returns:
            "tools": 승인됨 - 도구 실행
            "respond": 거부됨 - 응답 생성
        ====================================================================
        """
        if state.get("user_approved"):
            return "tools"
        return "respond"

    @traceable(name="generate_response")
    def _generate_response(self, state: AssistantState) -> dict:
        """
        ====================================================================
        최종 응답 생성 노드
        ====================================================================

        마지막 AI 메시지에서 응답을 추출하고 대화 기록에 저장합니다.

        Args:
            state: 현재 그래프 상태

        Returns:
            dict: 상태 업데이트 (response)
        ====================================================================
        """
        last_message = state["messages"][-1]

        # 응답 내용 추출
        if hasattr(last_message, "content"):
            response = last_message.content
        else:
            response = str(last_message)

        # 대화 기록에 추가
        session_id = state["session_id"]
        memory = conversation_store.get_session(session_id)
        memory.add_ai_message(response)

        return {"response": response}

    # ========================================================================
    # 그래프 컴파일 및 실행 메서드
    # ========================================================================

    def compile(self, with_memory: bool = True):
        """
        ====================================================================
        그래프 컴파일
        ====================================================================

        그래프를 실행 가능한 형태로 컴파일합니다.

        compile() 상세 설명:
        --------------------------------------------------------------------
        StateGraph의 compile() 메서드는 정의된 그래프를 실행 가능한 객체로 변환합니다.

        반환값:
            CompiledGraph 객체 - invoke(), stream(), ainvoke() 등의 메서드 제공

        주요 매개변수:
            - checkpointer: 체크포인터 인스턴스 (상태 저장용)
            - interrupt_before: 특정 노드 전에 실행 중단
            - interrupt_after: 특정 노드 후에 실행 중단

        MemorySaver 상세 설명:
        --------------------------------------------------------------------
        MemorySaver는 LangGraph에서 제공하는 인메모리 체크포인터입니다.

        체크포인터(Checkpointer)란?
            - 그래프 실행 상태를 저장하고 복원하는 객체
            - 대화 기록 유지, 실행 재개 등에 사용

        MemorySaver 특징:
            - 메모리에 상태 저장 (휘발성)
            - 개발/테스트에 적합
            - 프로덕션에서는 SqliteSaver, PostgresSaver 등 사용

        체크포인팅의 장점:
            1. 대화 연속성: 같은 thread_id로 대화 이어가기
            2. 시간 여행: 이전 상태로 돌아가기
            3. 분기: 특정 시점에서 다른 경로 탐색

        사용 예시:
            # 체크포인터로 컴파일
            app = graph.compile(checkpointer=MemorySaver())

            # thread_id로 대화 식별
            config = {"configurable": {"thread_id": "user-123"}}
            result = app.invoke(state, config=config)

            # 같은 thread_id로 대화 이어가기
            result2 = app.invoke(new_state, config=config)

        Args:
            with_memory: 체크포인터 사용 여부 (기본값: True)

        Returns:
            CompiledGraph: 컴파일된 그래프
        ====================================================================
        """
        if with_memory:
            checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=checkpointer)
        else:
            self.app = self.graph.compile()
        return self.app

    @traceable(name="chat")
    def chat(
        self,
        message: str,
        session_id: str = "default"
    ) -> str:
        """
        ====================================================================
        동기식 채팅
        ====================================================================

        사용자 메시지를 받아 응답을 생성합니다.

        invoke() 상세 설명:
        --------------------------------------------------------------------
        컴파일된 그래프를 동기적으로 실행합니다.

        매개변수:
            - input: 초기 상태 또는 입력 데이터
            - config: 실행 설정 (thread_id 등)
            - stream_mode: (invoke에서는 무시됨)

        config의 configurable:
            - thread_id: 대화 스레드 식별자
                같은 thread_id를 사용하면 체크포인터가 이전 상태를 복원

        반환값:
            - 최종 상태 (dict)

        Args:
            message: 사용자 입력 메시지
            session_id: 세션 ID (기본값: "default")

        Returns:
            str: AI 응답
        ====================================================================
        """
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(message, session_id)
        config = {"configurable": {"thread_id": session_id}}

        result = self.app.invoke(initial_state, config=config)

        return result["response"]

    async def achat(
        self,
        message: str,
        session_id: str = "default"
    ) -> str:
        """
        ====================================================================
        비동기식 채팅
        ====================================================================

        ainvoke() 상세 설명:
        --------------------------------------------------------------------
        컴파일된 그래프를 비동기적으로 실행합니다.
        invoke()의 비동기 버전으로, asyncio 이벤트 루프에서 실행됩니다.

        사용 시 주의:
            - async 함수 내에서만 사용 가능
            - await 키워드 필요

        사용 예시:
            async def main():
                response = await assistant.achat("안녕!")
                print(response)

            asyncio.run(main())

        Args:
            message: 사용자 입력 메시지
            session_id: 세션 ID (기본값: "default")

        Returns:
            str: AI 응답
        ====================================================================
        """
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(message, session_id)
        config = {"configurable": {"thread_id": session_id}}

        result = await self.app.ainvoke(initial_state, config=config)

        return result["response"]

    def stream_chat(
        self,
        message: str,
        session_id: str = "default"
    ):
        """
        ====================================================================
        스트리밍 채팅
        ====================================================================

        stream() 상세 설명:
        --------------------------------------------------------------------
        그래프 실행 결과를 스트리밍으로 반환합니다.
        각 노드 실행 후 상태를 yield합니다.

        매개변수:
            - input: 초기 상태
            - config: 실행 설정
            - stream_mode: 스트리밍 모드
                - "values": 각 노드 후 전체 상태 반환
                - "updates": 변경된 부분만 반환
                - "messages": 메시지만 스트리밍

        stream_mode 비교:
            # "values" 모드
            for state in app.stream(input, stream_mode="values"):
                # state = 전체 상태 딕셔너리
                print(state["messages"])

            # "updates" 모드
            for update in app.stream(input, stream_mode="updates"):
                # update = {"node_name": {변경된 필드들}}
                print(update)

        제너레이터 사용:
            - yield를 사용하여 결과를 순차적으로 반환
            - 호출자는 for 루프로 결과를 순회

        Args:
            message: 사용자 입력 메시지
            session_id: 세션 ID (기본값: "default")

        Yields:
            dict: 이벤트 정보
                - type: "message" 또는 "tool_call"
                - content/calls: 해당 데이터
        ====================================================================
        """
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(message, session_id)
        config = {"configurable": {"thread_id": session_id}}

        # stream_mode="values"로 각 노드 후 전체 상태 받기
        for event in self.app.stream(
            initial_state,
            config=config,
            stream_mode="values"
        ):
            if "messages" in event:
                last_msg = event["messages"][-1]

                # AI 응답 텍스트
                if hasattr(last_msg, "content") and last_msg.content:
                    yield {
                        "type": "message",
                        "content": last_msg.content
                    }

                # 도구 호출 정보
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    yield {
                        "type": "tool_call",
                        "calls": last_msg.tool_calls
                    }

    def approve_tool_execution(
        self,
        session_id: str,
        approved: bool = True
    ):
        """
        ====================================================================
        도구 실행 승인
        ====================================================================

        update_state() 상세 설명:
        --------------------------------------------------------------------
        컴파일된 그래프의 현재 상태를 업데이트합니다.
        체크포인터를 사용할 때 외부에서 상태를 수정할 수 있습니다.

        사용 시나리오:
            1. 그래프가 승인을 기다리며 중단됨
            2. 사용자가 UI에서 승인/거부 선택
            3. update_state로 승인 상태 업데이트
            4. 그래프 실행 재개

        매개변수:
            - config: thread_id 포함
            - values: 업데이트할 상태 값

        Args:
            session_id: 세션 ID
            approved: 승인 여부 (기본값: True)
        ====================================================================
        """
        if self.app is None:
            return

        config = {"configurable": {"thread_id": session_id}}

        # 승인 상태 업데이트
        self.app.update_state(
            config,
            {"user_approved": approved, "requires_approval": False}
        )

    def get_conversation_history(
        self,
        session_id: str
    ) -> List[Dict]:
        """
        ====================================================================
        대화 기록 가져오기
        ====================================================================

        Args:
            session_id: 세션 ID

        Returns:
            List[Dict]: 대화 기록 리스트
                [{"role": "user", "content": "..."}, ...]
        ====================================================================
        """
        memory = conversation_store.get_session(session_id)
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in memory.messages
        ]

    def clear_conversation(self, session_id: str):
        """
        ====================================================================
        대화 기록 초기화
        ====================================================================

        Args:
            session_id: 세션 ID
        ====================================================================
        """
        memory = conversation_store.get_session(session_id)
        memory.clear()


# ============================================================================
# 메인 실행 (CLI 모드)
# ============================================================================
if __name__ == "__main__":
    """
    직접 실행 시 CLI 채팅 모드로 동작합니다.

    사용법:
        python -m assistant.graph.assistant

    명령:
        - 텍스트 입력: AI에게 질문
        - quit/exit: 프로그램 종료
    """
    assistant = AIAssistant()

    print("AI 어시스턴트가 시작되었습니다. 'quit'로 종료합니다.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("안녕히 가세요!")
            break

        response = assistant.chat(user_input, session_id="cli-session")
        print(f"\nAssistant: {response}\n")
