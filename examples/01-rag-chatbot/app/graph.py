# app/graph.py
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.chains import RAGChains
from app.config import Config


class ChatState(TypedDict):
    """챗봇 상태"""
    messages: Annotated[list, add_messages]
    question: str
    query_type: str
    retrieved_docs: List[Document]
    context: str
    answer: str


class RAGChatGraph:
    """RAG 챗봇 그래프"""

    def __init__(self):
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=Config.TEMPERATURE)
        self.rag_chains = RAGChains()
        self.retriever = self.rag_chains.vs_manager.get_retriever()
        self.graph = self._build_graph()

    def _classify_query(self, state: ChatState) -> dict:
        """질문 분류 노드"""
        question = state["question"]

        router_prompt = ChatPromptTemplate.from_template(
            """질문을 분류하세요.
- "search": 정보 검색이 필요한 질문
- "chat": 일반 대화, 인사, 감사 등

질문: {question}

분류 (search/chat):"""
        )

        chain = router_prompt | self.llm
        result = chain.invoke({"question": question})
        query_type = result.content.strip().lower()

        if query_type not in ["search", "chat"]:
            query_type = "search"  # 기본값

        return {"query_type": query_type}

    def _retrieve_documents(self, state: ChatState) -> dict:
        """문서 검색 노드"""
        question = state["question"]
        messages = state.get("messages", [])

        # 대화 기록이 있으면 질문 재작성
        if len(messages) > 1:
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "대화 기록을 참고하여 독립적인 질문으로 재작성하세요."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            chain = contextualize_prompt | self.llm
            result = chain.invoke({
                "history": messages[:-1],
                "question": question
            })
            search_query = result.content
        else:
            search_query = question

        # 문서 검색
        docs = self.retriever.invoke(search_query)

        # 컨텍스트 포맷팅
        context = self._format_docs(docs)

        return {
            "retrieved_docs": docs,
            "context": context
        }

    def _generate_rag_answer(self, state: ChatState) -> dict:
        """RAG 기반 답변 생성 노드"""
        context = state["context"]
        question = state["question"]
        messages = state.get("messages", [])

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

        chain = rag_prompt | self.llm
        result = chain.invoke({
            "history": messages[:-1] if messages else [],
            "context": context,
            "question": question
        })

        answer = result.content
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }

    def _generate_chat_answer(self, state: ChatState) -> dict:
        """일반 대화 답변 생성 노드"""
        question = state["question"]
        messages = state.get("messages", [])

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

    def _route_query(self, state: ChatState) -> Literal["retrieve", "chat"]:
        """질문 유형에 따른 라우팅"""
        query_type = state.get("query_type", "search")
        if query_type == "chat":
            return "chat"
        return "retrieve"

    def _format_docs(self, docs: List[Document]) -> str:
        """문서 포맷팅"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _build_graph(self) -> StateGraph:
        """그래프 구축"""
        # 그래프 생성
        graph = StateGraph(ChatState)

        # 노드 추가
        graph.add_node("classify", self._classify_query)
        graph.add_node("retrieve", self._retrieve_documents)
        graph.add_node("rag_answer", self._generate_rag_answer)
        graph.add_node("chat_answer", self._generate_chat_answer)

        # 엣지 추가
        graph.add_edge(START, "classify")

        # 조건부 라우팅
        graph.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "retrieve": "retrieve",
                "chat": "chat_answer"
            }
        )

        graph.add_edge("retrieve", "rag_answer")
        graph.add_edge("rag_answer", END)
        graph.add_edge("chat_answer", END)

        return graph

    def compile(self, with_memory: bool = True):
        """그래프 컴파일"""
        if with_memory:
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
        │ classify │
        └────┬─────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  ┌────────┐   ┌────────────┐
  │retrieve│   │chat_answer │
  └───┬────┘   └─────┬──────┘
      │              │
      ▼              │
 ┌───────────┐       │
 │rag_answer │       │
 └─────┬─────┘       │
       │             │
       └──────┬──────┘
              │
              ▼
         ┌────────┐
         │  END   │
         └────────┘
        """


class RAGChatbot:
    """RAG 챗봇 인터페이스"""

    def __init__(self):
        self.graph = RAGChatGraph()
        self.app = self.graph.compile(with_memory=True)
        self.thread_id = "default"

    def set_thread(self, thread_id: str):
        """대화 스레드 설정"""
        self.thread_id = thread_id

    def chat(self, message: str) -> dict:
        """메시지 전송 및 응답 받기"""
        config = {"configurable": {"thread_id": self.thread_id}}

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
        """스트리밍 응답"""
        config = {"configurable": {"thread_id": self.thread_id}}

        for event in self.app.stream(
            {
                "question": message,
                "messages": [HumanMessage(content=message)]
            },
            config=config,
            stream_mode="updates"
        ):
            yield event


# 사용 예시
if __name__ == "__main__":
    chatbot = RAGChatbot()

    # 그래프 시각화
    print(chatbot.graph.visualize())

    # 대화 테스트
    print("\n=== 대화 테스트 ===")

    response = chatbot.chat("안녕하세요!")
    print(f"Bot: {response['answer']}")
    print(f"Type: {response['query_type']}")

    response = chatbot.chat("LangChain이 무엇인가요?")
    print(f"Bot: {response['answer']}")
    print(f"Type: {response['query_type']}")
    print(f"Sources: {response['sources']}")
