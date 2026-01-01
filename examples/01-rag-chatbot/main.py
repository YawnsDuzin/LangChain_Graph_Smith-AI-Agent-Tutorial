"""
RAG 챗봇 예제
============
LangChain, LangGraph, LangSmith를 활용한 문서 기반 Q&A 챗봇

사용법:
    python main.py
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ============ 설정 ============
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "gpt-4"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200


# ============ 상태 정의 ============
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    context: str
    answer: str


# ============ 샘플 문서 ============
SAMPLE_DOCUMENTS = [
    Document(
        page_content="""
LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 프레임워크입니다.
주요 기능으로는 체인, 에이전트, 메모리 관리, 도구 사용 등이 있습니다.
LangChain을 사용하면 복잡한 AI 워크플로우를 쉽게 구축할 수 있습니다.
""",
        metadata={"source": "langchain_intro.md"}
    ),
    Document(
        page_content="""
LangGraph는 LangChain 팀에서 개발한 그래프 기반 워크플로우 라이브러리입니다.
상태 기반(stateful) 멀티 액터 애플리케이션을 구축할 수 있으며,
순환 그래프, 체크포인팅, Human-in-the-Loop 등의 기능을 제공합니다.
""",
        metadata={"source": "langgraph_intro.md"}
    ),
    Document(
        page_content="""
LangSmith는 LLM 애플리케이션의 개발, 테스트, 배포, 모니터링을 위한 플랫폼입니다.
트레이싱을 통해 모든 LLM 호출을 추적할 수 있으며,
평가(Evaluation) 기능으로 모델 성능을 객관적으로 측정할 수 있습니다.
""",
        metadata={"source": "langsmith_intro.md"}
    ),
]


# ============ RAG 챗봇 클래스 ============
class RAGChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.7)
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = None
        self.retriever = None
        self.app = None

        self._setup_vectorstore()
        self._build_graph()

    def _setup_vectorstore(self):
        """벡터 스토어 설정"""
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(SAMPLE_DOCUMENTS)

        # 벡터 스토어 생성
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ 벡터 스토어 설정 완료")

    def _retrieve_context(self, state: ChatState) -> dict:
        """문서 검색"""
        question = state["question"]
        docs = self.retriever.invoke(question)

        context = "\n\n".join([
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

        return {"context": context}

    def _generate_answer(self, state: ChatState) -> dict:
        """답변 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 문서 기반 Q&A 어시스턴트입니다.
주어진 컨텍스트를 바탕으로 질문에 답변해주세요.
컨텍스트에 없는 정보는 "문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", """컨텍스트:
{context}

질문: {question}

답변:""")
        ])

        chain = prompt | self.llm | StrOutputParser()

        messages = state.get("messages", [])
        history = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

        answer = chain.invoke({
            "context": state["context"],
            "question": state["question"],
            "history": history[-6:]  # 최근 6개 메시지만
        })

        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }

    def _build_graph(self):
        """그래프 구축"""
        graph = StateGraph(ChatState)

        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("generate", self._generate_answer)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        checkpointer = MemorySaver()
        self.app = graph.compile(checkpointer=checkpointer)
        print("✅ 그래프 컴파일 완료")

    def chat(self, message: str, session_id: str = "default") -> str:
        """채팅"""
        config = {"configurable": {"thread_id": session_id}}

        result = self.app.invoke(
            {
                "question": message,
                "messages": [HumanMessage(content=message)]
            },
            config=config
        )

        return result["answer"]


# ============ 메인 ============
def main():
    print("\n" + "="*50)
    print("📚 RAG 문서 Q&A 챗봇")
    print("="*50)
    print("\n샘플 문서: LangChain, LangGraph, LangSmith 소개")
    print("종료하려면 'quit' 입력\n")

    chatbot = RAGChatbot()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "종료"]:
                print("👋 안녕히 가세요!")
                break

            response = chatbot.chat(user_input)
            print(f"\nBot: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


if __name__ == "__main__":
    main()
