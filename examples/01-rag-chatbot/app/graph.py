# app/graph.py
"""
LangGraph 기반 RAG 챗봇 그래프 모듈

이 모듈은 LangGraph를 사용하여 상태 기반 워크플로우를 구현합니다.
조건부 분기, 상태 관리, 체크포인팅을 통해 복잡한 대화 흐름을 구성합니다.

LangGraph 핵심 개념:
--------------------

1. StateGraph (상태 그래프)
   - 노드와 엣지로 구성된 방향 그래프
   - 각 노드는 상태를 변환하는 함수
   - 엣지는 노드 간 연결 (조건부/무조건부)

2. State (상태)
   - TypedDict로 정의된 상태 스키마
   - 그래프 실행 중 공유되는 데이터
   - 각 노드가 상태를 읽고 업데이트

3. Node (노드)
   - 상태를 입력받아 처리 후 반환하는 함수
   - 반환값으로 상태 업데이트 (부분 업데이트 가능)

4. Edge (엣지)
   - add_edge: 무조건적 연결
   - add_conditional_edges: 조건에 따른 분기

5. Checkpoint (체크포인트)
   - 실행 중간 상태 저장
   - 대화 기록, 상태 복원 가능
   - MemorySaver 또는 외부 저장소 사용

워크플로우 다이어그램:
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌──────────┐
    │ classify │ ← 질문 유형 분류
    └────┬─────┘
         │
  ┌──────┴──────┐
  │             │
  ▼             ▼
┌────────┐  ┌────────────┐
│retrieve│  │chat_answer │ ← 일반 대화
└───┬────┘  └─────┬──────┘
    │             │
    ▼             │
┌───────────┐     │
│rag_answer │     │ ← RAG 답변
└─────┬─────┘     │
      │           │
      └──────┬────┘
             │
             ▼
        ┌────────┐
        │  END   │
        └────────┘
"""

from typing import TypedDict, Annotated, List, Literal

# =============================================================================
# LangGraph 임포트 설명
# =============================================================================

# langgraph.graph.StateGraph
# ---------------------------
# 상태 기반 워크플로우 그래프 클래스입니다.
#
# 주요 메서드:
# - add_node(name, function): 노드 추가
# - add_edge(from_node, to_node): 무조건 엣지 추가
# - add_conditional_edges(node, condition_fn, mapping): 조건부 엣지
# - compile(): 실행 가능한 앱으로 변환
#
# 사용 예시:
#   graph = StateGraph(MyState)
#   graph.add_node("process", process_fn)
#   graph.add_edge(START, "process")
#   graph.add_edge("process", END)
#   app = graph.compile()
#   result = app.invoke(initial_state)
from langgraph.graph import StateGraph, START, END

# langgraph.graph.message.add_messages
# ------------------------------------
# 메시지 리스트를 업데이트하는 리듀서 함수입니다.
# Annotated와 함께 사용하여 상태 필드의 업데이트 방식을 정의합니다.
#
# 리듀서(Reducer)란?
# - 상태 업데이트 시 기존 값과 새 값을 어떻게 결합할지 정의
# - 예: 덮어쓰기, 추가, 병합 등
#
# add_messages 동작:
# - 새 메시지를 기존 메시지 리스트에 추가
# - 같은 ID의 메시지는 덮어쓰기
#
# 사용 예시:
#   class State(TypedDict):
#       messages: Annotated[list, add_messages]
#
#   # 노드에서 반환:
#   return {"messages": [new_message]}
#   # → 기존 messages에 new_message 추가됨
from langgraph.graph.message import add_messages

# langgraph.checkpoint.memory.MemorySaver
# ----------------------------------------
# 인메모리 체크포인터입니다.
# 그래프 실행 상태를 메모리에 저장합니다.
#
# 체크포인터의 역할:
# - 실행 중간 상태 저장
# - 대화 스레드별 상태 관리
# - 상태 복원 및 재개
#
# 프로덕션에서는 다른 옵션 사용:
# - PostgresSaver: PostgreSQL 저장
# - SQLiteSaver: SQLite 파일 저장
# - RedisSaver: Redis 저장
#
# 사용 예시:
#   checkpointer = MemorySaver()
#   app = graph.compile(checkpointer=checkpointer)
#   result = app.invoke(state, config={"configurable": {"thread_id": "1"}})
from langgraph.checkpoint.memory import MemorySaver

# LangChain 메시지 타입
# ---------------------
# LLM 대화에서 사용되는 메시지 객체들입니다.
#
# HumanMessage: 사용자 입력
# AIMessage: AI 응답
# SystemMessage: 시스템 지시사항
#
# 각 메시지 구조:
#   message.content: 텍스트 내용
#   message.type: 메시지 유형 ("human", "ai", "system")
#   message.id: 고유 식별자 (선택)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.chains import RAGChains
from app.config import Config


# =============================================================================
# 상태 정의 (State Schema)
# =============================================================================

class ChatState(TypedDict):
    """
    챗봇 상태 스키마

    TypedDict를 사용하여 상태의 구조를 정의합니다.
    그래프의 모든 노드가 이 상태를 공유합니다.

    TypedDict란?
    - Python의 타입 힌트 기능
    - 딕셔너리의 키와 값 타입을 명시
    - 런타임에는 일반 dict처럼 동작
    - IDE 자동완성 및 타입 체크 지원

    Annotated 필드:
    - Annotated[type, reducer]로 업데이트 방식 정의
    - messages 필드: add_messages 리듀서로 메시지 누적
    - 다른 필드: 기본적으로 덮어쓰기

    상태 업데이트 예시:
        # 노드 함수에서
        return {"query_type": "search"}  # query_type만 업데이트
        return {"messages": [AIMessage(...)]}  # messages에 추가
    """
    # messages: 대화 기록
    # Annotated[list, add_messages]는 반환된 메시지를 기존 리스트에 추가
    messages: Annotated[list, add_messages]

    # question: 현재 사용자 질문
    question: str

    # query_type: 질문 유형 ("search" 또는 "chat")
    query_type: str

    # retrieved_docs: 검색된 문서 청크 리스트
    retrieved_docs: List[Document]

    # context: 포맷팅된 문서 컨텍스트
    context: str

    # answer: 생성된 답변
    answer: str


class RAGChatGraph:
    """
    RAG 챗봇 그래프 클래스

    LangGraph StateGraph를 사용하여 RAG 워크플로우를 구현합니다.

    워크플로우:
    1. classify: 질문 유형 분류 (search/chat)
    2. 분기:
       - search → retrieve → rag_answer
       - chat → chat_answer
    3. END: 응답 반환

    사용 예시:
        graph = RAGChatGraph()
        app = graph.compile()
        result = app.invoke({
            "question": "LangChain이란?",
            "messages": [HumanMessage(content="LangChain이란?")]
        })
    """

    def __init__(self):
        """
        RAGChatGraph 초기화

        LLM, RAG 체인, 리트리버를 설정하고 그래프를 구축합니다.
        """
        # LLM 초기화
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=Config.TEMPERATURE)

        # RAG 체인 및 리트리버
        self.rag_chains = RAGChains()
        self.retriever = self.rag_chains.vs_manager.get_retriever()

        # 그래프 구축
        self.graph = self._build_graph()

    # =========================================================================
    # 노드 함수들 (Node Functions)
    # =========================================================================
    # 각 노드는 state를 입력받아 상태 업데이트 딕셔너리를 반환합니다.
    # 반환된 딕셔너리의 키만 상태에서 업데이트됩니다 (부분 업데이트).
    # =========================================================================

    def _classify_query(self, state: ChatState) -> dict:
        """
        질문 분류 노드

        사용자 질문이 문서 검색이 필요한지 판단합니다.

        Args:
            state (ChatState): 현재 상태

        Returns:
            dict: {"query_type": "search" or "chat"}

        노드 함수 규칙:
        - state 전체를 입력으로 받음
        - 필요한 필드만 업데이트하는 dict 반환
        - 반환하지 않은 필드는 변경되지 않음
        """
        question = state["question"]

        # 질문 분류 프롬프트
        router_prompt = ChatPromptTemplate.from_template(
            """질문을 분류하세요.
- "search": 정보 검색이 필요한 질문
- "chat": 일반 대화, 인사, 감사 등

질문: {question}

분류 (search/chat):"""
        )

        # LLM으로 분류
        # LCEL: prompt | llm
        chain = router_prompt | self.llm
        result = chain.invoke({"question": question})

        # 결과에서 분류값 추출
        query_type = result.content.strip().lower()

        # 유효하지 않은 값이면 기본값 사용
        if query_type not in ["search", "chat"]:
            query_type = "search"  # 기본값

        # 상태 업데이트: query_type만 변경
        return {"query_type": query_type}

    def _retrieve_documents(self, state: ChatState) -> dict:
        """
        문서 검색 노드

        질문을 기반으로 관련 문서를 검색합니다.
        대화 기록이 있으면 질문을 재작성합니다.

        Args:
            state (ChatState): 현재 상태

        Returns:
            dict: {"retrieved_docs": [...], "context": "..."}
        """
        question = state["question"]
        messages = state.get("messages", [])

        # 대화 기록이 있으면 질문 재작성
        # (이전 대화 맥락을 반영한 독립적인 질문으로 변환)
        if len(messages) > 1:
            # 질문 재작성 프롬프트
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "대화 기록을 참고하여 독립적인 질문으로 재작성하세요."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            chain = contextualize_prompt | self.llm
            result = chain.invoke({
                "history": messages[:-1],  # 마지막 메시지 제외
                "question": question
            })
            search_query = result.content
        else:
            search_query = question

        # =====================================================================
        # Retriever 검색
        # =====================================================================
        # retriever.invoke(query):
        # 1. query를 임베딩으로 변환
        # 2. 벡터 스토어에서 유사한 문서 검색
        # 3. List[Document] 반환
        #
        # 반환되는 문서 수는 retriever 생성 시 설정:
        #   vectorstore.as_retriever(search_kwargs={"k": 4})
        # =====================================================================
        docs = self.retriever.invoke(search_query)

        # 검색된 문서를 문자열로 포맷팅
        context = self._format_docs(docs)

        return {
            "retrieved_docs": docs,
            "context": context
        }

    def _generate_rag_answer(self, state: ChatState) -> dict:
        """
        RAG 기반 답변 생성 노드

        검색된 문서와 대화 기록을 바탕으로 답변을 생성합니다.

        Args:
            state (ChatState): 현재 상태

        Returns:
            dict: {"answer": "...", "messages": [AIMessage(...)]}
        """
        context = state["context"]
        question = state["question"]
        messages = state.get("messages", [])

        # RAG 프롬프트
        # 시스템 메시지 + 대화 기록 + 컨텍스트/질문
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 문서 기반 Q&A 어시스턴트입니다.
주어진 컨텍스트를 바탕으로 질문에 답변해주세요.

규칙:
1. 컨텍스트에 있는 정보만 사용하세요.
2. 정보가 없으면 솔직히 모른다고 하세요.
3. 출처가 있으면 언급하세요."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", """컨텍스트:
{context}

질문: {question}

답변:""")
        ])

        # LLM으로 답변 생성
        chain = rag_prompt | self.llm
        result = chain.invoke({
            "history": messages[:-1] if messages else [],
            "context": context,
            "question": question
        })

        answer = result.content

        # 상태 업데이트
        # messages에 AIMessage 추가 (add_messages 리듀서 적용)
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }

    def _generate_chat_answer(self, state: ChatState) -> dict:
        """
        일반 대화 답변 생성 노드

        문서 검색 없이 일반적인 대화 응답을 생성합니다.

        Args:
            state (ChatState): 현재 상태

        Returns:
            dict: {"answer": "...", "messages": [AIMessage(...)]}
        """
        question = state["question"]
        messages = state.get("messages", [])

        # 일반 대화 프롬프트
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 친절한 AI 어시스턴트입니다.
자연스럽고 친근하게 대화해주세요."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = chat_prompt | self.llm
        result = chain.invoke({
            "history": messages[:-1] if messages else [],
            "question": question
        })

        answer = result.content
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }

    # =========================================================================
    # 조건부 라우팅 함수
    # =========================================================================

    def _route_query(self, state: ChatState) -> Literal["retrieve", "chat"]:
        """
        질문 유형에 따른 라우팅

        classify 노드 이후 다음 노드를 결정합니다.

        Args:
            state (ChatState): 현재 상태

        Returns:
            Literal["retrieve", "chat"]: 다음 노드 키

        조건부 라우팅 규칙:
        - 반환값은 add_conditional_edges의 mapping 키와 일치해야 함
        - 예: "retrieve" → "retrieve" 노드로 이동
        """
        query_type = state.get("query_type", "search")
        if query_type == "chat":
            return "chat"
        return "retrieve"

    def _format_docs(self, docs: List[Document]) -> str:
        """문서 포맷팅 유틸리티"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    # =========================================================================
    # 그래프 구축
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        """
        그래프 구축

        노드와 엣지를 추가하여 워크플로우를 정의합니다.

        Returns:
            StateGraph: 구축된 그래프 (compile 전)

        그래프 구축 단계:
        1. StateGraph 생성 (상태 스키마 지정)
        2. 노드 추가 (add_node)
        3. 엣지 추가 (add_edge, add_conditional_edges)
        """
        # =====================================================================
        # StateGraph 생성
        # =====================================================================
        # StateGraph(StateType)
        # - StateType: 상태 스키마 (TypedDict)
        # - 이 스키마에 정의된 필드만 상태에 포함 가능
        # =====================================================================
        graph = StateGraph(ChatState)

        # =====================================================================
        # 노드 추가 (add_node)
        # =====================================================================
        # graph.add_node(name, function)
        # - name: 노드 식별자 (문자열)
        # - function: 상태를 처리하는 함수 (state -> dict)
        #
        # 노드 함수 시그니처:
        #   def node_fn(state: StateType) -> dict:
        #       # 처리 로직
        #       return {"key": new_value}  # 상태 업데이트
        # =====================================================================
        graph.add_node("classify", self._classify_query)      # 질문 분류
        graph.add_node("retrieve", self._retrieve_documents)  # 문서 검색
        graph.add_node("rag_answer", self._generate_rag_answer)  # RAG 답변
        graph.add_node("chat_answer", self._generate_chat_answer)  # 일반 답변

        # =====================================================================
        # 엣지 추가 (add_edge)
        # =====================================================================
        # graph.add_edge(from_node, to_node)
        # - from_node: 시작 노드 (또는 START)
        # - to_node: 도착 노드 (또는 END)
        #
        # 특수 노드:
        # - START: 그래프 시작점 (진입점)
        # - END: 그래프 종료점 (출구점)
        # =====================================================================
        graph.add_edge(START, "classify")  # 시작 → 분류

        # =====================================================================
        # 조건부 엣지 추가 (add_conditional_edges)
        # =====================================================================
        # graph.add_conditional_edges(
        #     source_node,     # 분기 시작 노드
        #     condition_fn,    # 조건 판단 함수 (state -> str)
        #     mapping          # 반환값 → 노드 매핑
        # )
        #
        # condition_fn:
        # - 상태를 입력받아 문자열 반환
        # - 반환값이 mapping의 키와 일치
        #
        # mapping:
        # - {조건값: 다음노드} 딕셔너리
        # - 조건값이 다음 노드 결정
        #
        # 예시:
        #   def route(state):
        #       return "yes" if state["flag"] else "no"
        #
        #   graph.add_conditional_edges(
        #       "node_a",
        #       route,
        #       {"yes": "node_b", "no": "node_c"}
        #   )
        # =====================================================================
        graph.add_conditional_edges(
            "classify",        # 분류 노드에서
            self._route_query, # 라우팅 함수 적용
            {
                "retrieve": "retrieve",    # "retrieve" → retrieve 노드
                "chat": "chat_answer"      # "chat" → chat_answer 노드
            }
        )

        # retrieve → rag_answer 연결
        graph.add_edge("retrieve", "rag_answer")

        # 최종 노드들 → END 연결
        graph.add_edge("rag_answer", END)
        graph.add_edge("chat_answer", END)

        return graph

    def compile(self, with_memory: bool = True):
        """
        그래프 컴파일

        구축된 그래프를 실행 가능한 앱으로 변환합니다.

        Args:
            with_memory (bool): 체크포인터 사용 여부

        Returns:
            CompiledGraph: 실행 가능한 그래프

        compile() 메서드 설명:
        -----------------------
        - 그래프를 검증하고 최적화
        - 실행 가능한 Runnable 객체 반환
        - checkpointer 옵션으로 상태 저장 설정

        checkpointer (체크포인터):
        - 실행 중간 상태를 저장
        - thread_id로 대화 스레드 구분
        - 대화 기록 유지 및 복원

        사용 예시:
            app = graph.compile(checkpointer=MemorySaver())

            # 첫 번째 대화
            result1 = app.invoke(
                {"question": "안녕"},
                config={"configurable": {"thread_id": "user-1"}}
            )

            # 같은 스레드에서 이어서 대화
            result2 = app.invoke(
                {"question": "그러면?"},
                config={"configurable": {"thread_id": "user-1"}}
            )
        """
        if with_memory:
            # MemorySaver: 인메모리 체크포인터
            # 프로세스 종료 시 데이터 손실
            checkpointer = MemorySaver()
            return self.graph.compile(checkpointer=checkpointer)
        return self.graph.compile()

    def visualize(self) -> str:
        """그래프 시각화 (ASCII)"""
        return """
        ┌─────────┐
        │  START  │
        └────┬────┘
             │
             ▼
        ┌──────────┐
        │ classify │ ← 질문 분류
        └────┬─────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  ┌────────┐   ┌────────────┐
  │retrieve│   │chat_answer │ ← 일반 대화
  └───┬────┘   └─────┬──────┘
      │              │
      ▼              │
 ┌───────────┐       │
 │rag_answer │       │ ← RAG 답변
 └─────┬─────┘       │
       │             │
       └──────┬──────┘
              │
              ▼
         ┌────────┐
         │  END   │
         └────────┘
        """


# =============================================================================
# RAG 챗봇 인터페이스
# =============================================================================

class RAGChatbot:
    """
    RAG 챗봇 인터페이스 클래스

    RAGChatGraph를 래핑하여 더 간단한 API를 제공합니다.

    사용 예시:
        chatbot = RAGChatbot()

        # 기본 대화
        response = chatbot.chat("LangChain이란?")
        print(response["answer"])

        # 스트리밍
        for event in chatbot.stream_chat("LangGraph 설명해줘"):
            print(event)
    """

    def __init__(self):
        """챗봇 초기화"""
        self.graph = RAGChatGraph()
        self.app = self.graph.compile(with_memory=True)
        self.thread_id = "default"

    def set_thread(self, thread_id: str):
        """
        대화 스레드 설정

        다른 스레드 ID를 사용하면 별도의 대화 기록이 유지됩니다.

        Args:
            thread_id (str): 스레드 식별자
        """
        self.thread_id = thread_id

    def chat(self, message: str) -> dict:
        """
        메시지 전송 및 응답 받기

        Args:
            message (str): 사용자 메시지

        Returns:
            dict: {
                "answer": 답변 텍스트,
                "query_type": 질문 유형,
                "sources": 출처 리스트
            }

        invoke() 메서드 설명:
        ----------------------
        컴파일된 그래프를 실행합니다.

        파라미터:
        - 첫 번째 인자: 입력 상태 (dict)
        - config: 설정 딕셔너리
          - configurable: 런타임 설정
            - thread_id: 대화 스레드 ID

        반환값:
        - 최종 상태 (dict)
        - 모든 상태 필드 포함
        """
        # config: 스레드 ID로 대화 컨텍스트 관리
        config = {"configurable": {"thread_id": self.thread_id}}

        # =====================================================================
        # app.invoke() 상세 설명
        # =====================================================================
        # invoke(input, config) 메서드:
        # - input: 초기 상태 값 (dict)
        # - config: 실행 설정 (dict)
        #
        # 실행 흐름:
        # 1. 입력으로 초기 상태 설정
        # 2. START 노드에서 시작
        # 3. 각 노드 순차적으로 실행
        # 4. END 도달 시 종료
        # 5. 최종 상태 반환
        #
        # 체크포인터가 있으면:
        # - thread_id로 이전 상태 로드
        # - 대화 기록 이어서 진행
        # - 실행 후 상태 저장
        # =====================================================================
        result = self.app.invoke(
            {
                "question": message,
                "messages": [HumanMessage(content=message)]
            },
            config=config
        )

        return {
            "answer": result["answer"],
            "query_type": result.get("query_type", "unknown"),
            "sources": [
                doc.metadata.get("source", "Unknown")
                for doc in result.get("retrieved_docs", [])
            ]
        }

    def stream_chat(self, message: str):
        """
        스트리밍 응답

        노드 실행 결과를 실시간으로 반환합니다.

        Args:
            message (str): 사용자 메시지

        Yields:
            dict: 각 노드의 실행 결과

        stream() 메서드 설명:
        ----------------------
        그래프 실행을 스트리밍합니다.

        stream_mode 옵션:
        - "values": 각 단계의 전체 상태 반환
        - "updates": 각 노드가 업데이트한 부분만 반환
        - "debug": 디버그 정보 포함

        사용 예시:
            for event in app.stream(input, stream_mode="updates"):
                node_name = list(event.keys())[0]
                updates = event[node_name]
                print(f"{node_name}: {updates}")
        """
        config = {"configurable": {"thread_id": self.thread_id}}

        # =====================================================================
        # app.stream() 상세 설명
        # =====================================================================
        # stream(input, config, stream_mode) 메서드:
        # - Generator로 각 노드 결과를 순차적으로 반환
        #
        # stream_mode="updates":
        # - 각 노드가 상태에 적용한 업데이트만 반환
        # - 형식: {node_name: {updated_fields}}
        #
        # 실시간 UI 업데이트에 유용:
        # - 긴 작업의 진행 상황 표시
        # - 노드별 결과 즉시 표시
        # =====================================================================
        for event in self.app.stream(
            {
                "question": message,
                "messages": [HumanMessage(content=message)]
            },
            config=config,
            stream_mode="updates"  # 업데이트만 반환
        ):
            yield event


# =============================================================================
# 사용 예시 및 테스트
# =============================================================================
if __name__ == "__main__":
    """
    그래프 테스트

    실행 방법:
        python -m app.graph
    """
    chatbot = RAGChatbot()

    # 그래프 시각화
    print(chatbot.graph.visualize())

    # 대화 테스트
    print("\n=== 대화 테스트 ===")

    # 일반 대화 테스트
    print("\n[테스트 1: 일반 대화]")
    response = chatbot.chat("안녕하세요!")
    print(f"Bot: {response['answer']}")
    print(f"Type: {response['query_type']}")

    # RAG 질문 테스트
    print("\n[테스트 2: RAG 질문]")
    response = chatbot.chat("LangChain이 무엇인가요?")
    print(f"Bot: {response['answer']}")
    print(f"Type: {response['query_type']}")
    if response['sources']:
        print(f"Sources: {', '.join(response['sources'])}")
