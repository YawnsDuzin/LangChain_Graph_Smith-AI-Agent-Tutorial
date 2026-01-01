# 튜토리얼 1: RAG 기반 문서 Q&A 챗봇 구축

## 개요

이 튜토리얼에서는 **RAG(Retrieval-Augmented Generation)** 기법을 사용하여 문서 기반 Q&A 챗봇을 구축합니다. LangChain, LangGraph, LangSmith를 모두 활용하여 실제 프로덕션에서 사용할 수 있는 수준의 시스템을 만들어봅니다.

### 학습 목표

- PDF/텍스트 문서를 로드하고 처리하는 방법
- 벡터 스토어를 사용한 의미론적 검색 구현
- LangGraph로 대화형 RAG 시스템 구축
- LangSmith로 성능 모니터링 및 평가

### 최종 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG 챗봇 아키텍처                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   사용자 질문                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────┐                                           │
│   │   질문 분석      │ ◄──── 대화 기록 참조                       │
│   │  (Query Router) │                                           │
│   └────────┬────────┘                                           │
│            │                                                    │
│    ┌───────┴───────┐                                           │
│    │               │                                           │
│    ▼               ▼                                           │
│ ┌──────────┐  ┌──────────┐                                     │
│ │ 문서 검색  │  │ 일반 대화  │                                     │
│ │(Retrieval)│  │ (Chat)   │                                     │
│ └────┬─────┘  └────┬─────┘                                     │
│      │             │                                           │
│      ▼             │                                           │
│ ┌──────────┐       │                                           │
│ │  관련성   │       │                                           │
│ │  평가    │       │                                           │
│ └────┬─────┘       │                                           │
│      │             │                                           │
│      ▼             │                                           │
│ ┌──────────┐       │                                           │
│ │  답변    │◄──────┘                                           │
│ │  생성    │                                                    │
│ └────┬─────┘                                                   │
│      │                                                         │
│      ▼                                                         │
│ ┌──────────────────┐                                           │
│ │   LangSmith      │                                           │
│ │   모니터링        │                                           │
│ └──────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1단계: 프로젝트 설정

### 1.1 디렉토리 구조

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py          # 설정 관리
│   ├── document_loader.py # 문서 로딩
│   ├── vectorstore.py     # 벡터 스토어 관리
│   ├── chains.py          # LangChain 체인
│   ├── graph.py           # LangGraph 워크플로우
│   └── main.py            # 메인 애플리케이션
├── data/
│   └── documents/         # 원본 문서
├── vectordb/              # 벡터 DB 저장소
├── tests/
│   ├── test_retrieval.py
│   └── test_chains.py
├── .env
├── requirements.txt
└── README.md
```

### 1.2 의존성 설치

```bash
# requirements.txt
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langgraph>=0.1.0
langsmith>=0.1.0
chromadb>=0.4.0
pypdf>=4.0.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
```

```bash
pip install -r requirements.txt
```

### 1.3 환경 변수 설정

```bash
# .env
OPENAI_API_KEY=sk-your-openai-api-key
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-chatbot-tutorial
```

---

## 2단계: 설정 모듈 구현

### 2.1 config.py

```python
# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    """애플리케이션 설정"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "documents"
    VECTORDB_DIR = BASE_DIR / "vectordb"

    # Model Settings
    LLM_MODEL = "gpt-4"
    EMBEDDING_MODEL = "text-embedding-3-small"
    TEMPERATURE = 0.7

    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 4

    # LangSmith
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-chatbot")

    @classmethod
    def validate(cls):
        """설정 검증"""
        assert cls.OPENAI_API_KEY, "OPENAI_API_KEY가 설정되지 않았습니다."
        print("✅ 설정 검증 완료")

# 설정 검증
Config.validate()
```

---

## 3단계: 문서 로딩 및 처리

### 3.1 document_loader.py

```python
# app/document_loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config


class DocumentProcessor:
    """문서 로딩 및 처리 클래스"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """PDF 파일 로딩"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self._add_metadata(documents, file_path)

    def load_text(self, file_path: str) -> List[Document]:
        """텍스트 파일 로딩"""
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        return self._add_metadata(documents, file_path)

    def load_markdown(self, file_path: str) -> List[Document]:
        """마크다운 파일 로딩"""
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        return self._add_metadata(documents, file_path)

    def load_directory(self, directory_path: str = None) -> List[Document]:
        """디렉토리의 모든 문서 로딩"""
        if directory_path is None:
            directory_path = str(Config.DATA_DIR)

        all_documents = []
        path = Path(directory_path)

        # PDF 파일
        for pdf_file in path.glob("**/*.pdf"):
            docs = self.load_pdf(str(pdf_file))
            all_documents.extend(docs)

        # 텍스트 파일
        for txt_file in path.glob("**/*.txt"):
            docs = self.load_text(str(txt_file))
            all_documents.extend(docs)

        # 마크다운 파일
        for md_file in path.glob("**/*.md"):
            docs = self.load_markdown(str(md_file))
            all_documents.extend(docs)

        print(f"📄 총 {len(all_documents)}개 문서 로드됨")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        splits = self.text_splitter.split_documents(documents)
        print(f"📝 총 {len(splits)}개 청크로 분할됨")
        return splits

    def _add_metadata(self, documents: List[Document], file_path: str) -> List[Document]:
        """문서에 메타데이터 추가"""
        file_name = Path(file_path).name
        for doc in documents:
            doc.metadata["source"] = file_name
            doc.metadata["file_path"] = file_path
        return documents

    def process_documents(self, directory_path: str = None) -> List[Document]:
        """전체 문서 처리 파이프라인"""
        documents = self.load_directory(directory_path)
        splits = self.split_documents(documents)
        return splits


# 사용 예시
if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    print(f"처리된 청크 수: {len(chunks)}")
```

---

## 4단계: 벡터 스토어 구현

### 4.1 vectorstore.py

```python
# app/vectorstore.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from app.config import Config


class VectorStoreManager:
    """벡터 스토어 관리 클래스"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL
        )
        self.vectorstore: Optional[Chroma] = None
        self.persist_directory = str(Config.VECTORDB_DIR)

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """새 벡터 스토어 생성"""
        print("🔄 벡터 스토어 생성 중...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="rag_collection"
        )

        print(f"✅ 벡터 스토어 생성 완료: {len(documents)}개 문서")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """기존 벡터 스토어 로드"""
        print("🔄 벡터 스토어 로드 중...")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        print("✅ 벡터 스토어 로드 완료")
        return self.vectorstore

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = None
    ) -> BaseRetriever:
        """리트리버 생성"""
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K

        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        return retriever

    def similarity_search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """유사도 검색"""
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K
        results = self.vectorstore.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """점수와 함께 유사도 검색"""
        if self.vectorstore is None:
            self.load_vectorstore()

        k = k or Config.RETRIEVAL_K
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def add_documents(self, documents: List[Document]) -> None:
        """문서 추가"""
        if self.vectorstore is None:
            self.load_vectorstore()

        self.vectorstore.add_documents(documents)
        print(f"✅ {len(documents)}개 문서 추가됨")

    def delete_collection(self) -> None:
        """컬렉션 삭제"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print("🗑️ 컬렉션 삭제됨")


# 사용 예시
if __name__ == "__main__":
    from app.document_loader import DocumentProcessor

    # 문서 처리
    processor = DocumentProcessor()
    chunks = processor.process_documents()

    # 벡터 스토어 생성
    vs_manager = VectorStoreManager()
    vs_manager.create_vectorstore(chunks)

    # 검색 테스트
    results = vs_manager.similarity_search("LangChain이란?")
    for doc in results:
        print(f"- {doc.page_content[:100]}...")
```

---

## 5단계: LangChain 체인 구현

### 5.1 chains.py

```python
# app/chains.py
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from app.config import Config
from app.vectorstore import VectorStoreManager


class RAGChains:
    """RAG 관련 체인 클래스"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.vs_manager = VectorStoreManager()
        self.retriever = self.vs_manager.get_retriever()

    def _format_docs(self, docs: List[Document]) -> str:
        """문서들을 문자열로 포맷팅"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def create_rag_chain(self):
        """기본 RAG 체인 생성"""

        # RAG 프롬프트
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 문서 기반 Q&A 어시스턴트입니다.
주어진 컨텍스트를 바탕으로 질문에 정확하게 답변해주세요.

규칙:
1. 컨텍스트에 있는 정보만 사용하세요.
2. 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
3. 가능하면 출처를 언급하세요.
4. 답변은 명확하고 간결하게 작성하세요."""),
            ("human", """컨텍스트:
{context}

질문: {question}

답변:""")
        ])

        # 체인 구성
        chain = (
            {
                "context": self.retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def create_conversational_rag_chain(self):
        """대화형 RAG 체인 생성 (히스토리 포함)"""

        # 질문 재작성 프롬프트
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """대화 기록을 참고하여 사용자의 최신 질문을 재작성하세요.
대화 기록 없이도 이해할 수 있는 독립적인 질문으로 만드세요.
질문을 재작성할 필요가 없으면 그대로 반환하세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # 질문 재작성 체인
        contextualize_chain = (
            contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        # 대화형 RAG 프롬프트
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 친절한 문서 기반 Q&A 어시스턴트입니다.
주어진 컨텍스트와 대화 기록을 바탕으로 답변해주세요.

규칙:
1. 컨텍스트에 있는 정보를 우선 사용하세요.
2. 이전 대화 내용을 참고하여 자연스럽게 답변하세요.
3. 정보가 없으면 모른다고 솔직히 말하세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """컨텍스트:
{context}

질문: {question}

답변:""")
        ])

        def get_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """컨텍스트 검색"""
            question = inputs["question"]
            chat_history = inputs.get("chat_history", [])

            # 대화 기록이 있으면 질문 재작성
            if chat_history:
                standalone_question = contextualize_chain.invoke({
                    "question": question,
                    "chat_history": chat_history
                })
            else:
                standalone_question = question

            # 문서 검색
            docs = self.retriever.invoke(standalone_question)
            context = self._format_docs(docs)

            return {
                "context": context,
                "question": question,
                "chat_history": chat_history,
                "retrieved_docs": docs
            }

        # 대화형 RAG 체인
        chain = (
            RunnableLambda(get_context)
            | {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "retrieved_docs": lambda x: x["retrieved_docs"]
            }
            | RunnablePassthrough.assign(
                answer=qa_prompt | self.llm | StrOutputParser()
            )
        )

        return chain

    def create_query_router(self):
        """질문 라우터 생성 - 문서 검색이 필요한지 판단"""

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 질문 분류기입니다.
사용자의 질문이 문서 검색이 필요한 질문인지 판단하세요.

분류:
- "search": 특정 정보나 지식이 필요한 질문 (예: "LangChain이란?", "API 사용법을 알려줘")
- "chat": 일반적인 대화나 인사 (예: "안녕", "고마워", "잘 모르겠어")

오직 "search" 또는 "chat"만 응답하세요."""),
            ("human", "{question}")
        ])

        router_chain = (
            router_prompt
            | self.llm
            | StrOutputParser()
        )

        return router_chain


# 사용 예시
if __name__ == "__main__":
    chains = RAGChains()

    # 기본 RAG 테스트
    rag_chain = chains.create_rag_chain()
    result = rag_chain.invoke("LangChain의 주요 기능은?")
    print(f"답변: {result}")
```

---

## 6단계: LangGraph 워크플로우 구현

### 6.1 graph.py

```python
# app/graph.py
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
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
```

---

## 7단계: 메인 애플리케이션

### 7.1 main.py

```python
# app/main.py
import os
from pathlib import Path
from app.config import Config
from app.document_loader import DocumentProcessor
from app.vectorstore import VectorStoreManager
from app.graph import RAGChatbot


def initialize_vectorstore(force_rebuild: bool = False):
    """벡터 스토어 초기화"""
    vs_manager = VectorStoreManager()

    # 기존 벡터 스토어 확인
    if Config.VECTORDB_DIR.exists() and not force_rebuild:
        try:
            vs_manager.load_vectorstore()
            print("✅ 기존 벡터 스토어 로드됨")
            return vs_manager
        except Exception as e:
            print(f"⚠️ 벡터 스토어 로드 실패: {e}")

    # 새로 생성
    print("🔄 벡터 스토어 새로 생성 중...")

    # 문서 처리
    processor = DocumentProcessor()
    chunks = processor.process_documents()

    if not chunks:
        print("⚠️ 처리할 문서가 없습니다.")
        print(f"   문서를 {Config.DATA_DIR} 디렉토리에 추가하세요.")
        return None

    # 벡터 스토어 생성
    vs_manager.create_vectorstore(chunks)

    return vs_manager


def run_interactive_chat():
    """대화형 챗봇 실행"""
    print("\n" + "="*50)
    print("📚 RAG 문서 Q&A 챗봇")
    print("="*50)
    print("질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("-"*50 + "\n")

    chatbot = RAGChatbot()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "종료"]:
                print("👋 안녕히 가세요!")
                break

            # 응답 생성
            response = chatbot.chat(user_input)

            print(f"\nBot: {response['answer']}")

            if response.get("sources"):
                print(f"📄 출처: {', '.join(response['sources'])}")

            print()

        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def main():
    """메인 함수"""
    print("🚀 RAG 챗봇 시작...")

    # 설정 검증
    Config.validate()

    # 벡터 스토어 초기화
    vs_manager = initialize_vectorstore()

    if vs_manager is None:
        # 샘플 문서 생성
        create_sample_documents()
        vs_manager = initialize_vectorstore(force_rebuild=True)

    # 대화형 챗봇 실행
    run_interactive_chat()


def create_sample_documents():
    """샘플 문서 생성"""
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    sample_content = """# LangChain 소개

LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 프레임워크입니다.

## 주요 기능

1. **모델 통합**: OpenAI, Anthropic 등 다양한 LLM을 지원합니다.
2. **체인**: 여러 컴포넌트를 연결하여 복잡한 워크플로우를 구성합니다.
3. **에이전트**: 도구를 사용하여 동적으로 작업을 수행합니다.
4. **메모리**: 대화 기록을 관리합니다.

## 설치 방법

```bash
pip install langchain langchain-openai
```

## 간단한 예시

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4")
response = chat.invoke("안녕하세요!")
print(response.content)
```

# LangGraph 소개

LangGraph는 LangChain 기반의 그래프 워크플로우 라이브러리입니다.

## 특징

- 상태 기반 워크플로우
- 순환 그래프 지원
- 체크포인팅 기능
- 멀티 에이전트 지원

# LangSmith 소개

LangSmith는 LLM 애플리케이션의 모니터링 및 평가 플랫폼입니다.

## 주요 기능

- 트레이싱: 모든 LLM 호출 추적
- 평가: 자동화된 품질 평가
- 프롬프트 관리: 버전 관리 및 공유
- 데이터셋: 테스트 데이터 관리
"""

    sample_file = Config.DATA_DIR / "langchain_guide.md"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_content)

    print(f"📄 샘플 문서 생성됨: {sample_file}")


if __name__ == "__main__":
    main()
```

---

## 8단계: LangSmith 모니터링 설정

### 8.1 monitoring.py

```python
# app/monitoring.py
from langsmith import Client
from langsmith import traceable
from datetime import datetime, timedelta
from typing import Optional


class RAGMonitor:
    """RAG 시스템 모니터링"""

    def __init__(self, project_name: str = None):
        self.client = Client()
        self.project_name = project_name or "rag-chatbot-tutorial"

    def get_recent_runs(self, hours: int = 24, limit: int = 100):
        """최근 런 조회"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),
            is_root=True,
            limit=limit
        ))
        return runs

    def get_error_runs(self, hours: int = 24):
        """에러 런 조회"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),
            is_root=True,
            error=True
        ))
        return runs

    def calculate_metrics(self, hours: int = 24):
        """메트릭 계산"""
        runs = self.get_recent_runs(hours)

        if not runs:
            return None

        # 지연시간 계산
        latencies = []
        costs = []
        errors = 0

        for run in runs:
            if run.end_time and run.start_time:
                latency = (run.end_time - run.start_time).total_seconds()
                latencies.append(latency)

            if run.total_cost:
                costs.append(run.total_cost)

            if run.error:
                errors += 1

        return {
            "total_runs": len(runs),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "total_cost": sum(costs),
            "error_count": errors,
            "error_rate": errors / len(runs) if runs else 0
        }

    def print_report(self, hours: int = 24):
        """리포트 출력"""
        metrics = self.calculate_metrics(hours)

        if not metrics:
            print("데이터가 없습니다.")
            return

        print("\n" + "="*50)
        print(f"📊 RAG 챗봇 모니터링 리포트 (최근 {hours}시간)")
        print("="*50)
        print(f"총 런 수: {metrics['total_runs']}")
        print(f"평균 지연시간: {metrics['avg_latency']:.2f}초")
        print(f"최대 지연시간: {metrics['max_latency']:.2f}초")
        print(f"최소 지연시간: {metrics['min_latency']:.2f}초")
        print(f"총 비용: ${metrics['total_cost']:.4f}")
        print(f"에러 수: {metrics['error_count']}")
        print(f"에러율: {metrics['error_rate']:.1%}")
        print("="*50 + "\n")

    def add_feedback(self, run_id: str, score: float, comment: str = None):
        """피드백 추가"""
        self.client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=score,
            comment=comment
        )


# 사용 예시
if __name__ == "__main__":
    monitor = RAGMonitor()
    monitor.print_report(hours=24)
```

---

## 9단계: 테스트 및 평가

### 9.1 tests/test_rag.py

```python
# tests/test_rag.py
import pytest
from langsmith import Client, evaluate
from app.graph import RAGChatbot
from app.chains import RAGChains


class TestRAGChatbot:
    """RAG 챗봇 테스트"""

    @pytest.fixture
    def chatbot(self):
        return RAGChatbot()

    def test_general_chat(self, chatbot):
        """일반 대화 테스트"""
        response = chatbot.chat("안녕하세요!")
        assert "answer" in response
        assert response["query_type"] == "chat"

    def test_rag_query(self, chatbot):
        """RAG 쿼리 테스트"""
        response = chatbot.chat("LangChain이 무엇인가요?")
        assert "answer" in response
        assert response["query_type"] == "search"

    def test_conversation_memory(self, chatbot):
        """대화 메모리 테스트"""
        chatbot.set_thread("test-thread")

        # 첫 번째 질문
        response1 = chatbot.chat("제 이름은 홍길동입니다.")

        # 두 번째 질문 (기억 테스트)
        response2 = chatbot.chat("제 이름이 뭐라고 했죠?")
        assert "홍길동" in response2["answer"]


def create_evaluation_dataset():
    """평가 데이터셋 생성"""
    client = Client()

    dataset = client.create_dataset(
        dataset_name="rag-chatbot-eval",
        description="RAG 챗봇 평가 데이터셋"
    )

    examples = [
        {
            "inputs": {"question": "LangChain이란?"},
            "outputs": {"answer": "LLM 기반 애플리케이션 개발 프레임워크"}
        },
        {
            "inputs": {"question": "LangGraph의 특징은?"},
            "outputs": {"answer": "상태 기반 그래프 워크플로우"}
        },
        {
            "inputs": {"question": "LangSmith의 주요 기능은?"},
            "outputs": {"answer": "트레이싱, 평가, 모니터링"}
        }
    ]

    for example in examples:
        client.create_example(
            inputs=example["inputs"],
            outputs=example["outputs"],
            dataset_id=dataset.id
        )

    return dataset


def run_evaluation():
    """평가 실행"""
    chatbot = RAGChatbot()

    def predict(inputs: dict) -> dict:
        response = chatbot.chat(inputs["question"])
        return {"answer": response["answer"]}

    def relevance_evaluator(run, example) -> dict:
        """관련성 평가"""
        from langchain_openai import ChatOpenAI

        judge = ChatOpenAI(model="gpt-4", temperature=0)
        question = example.inputs["question"]
        answer = run.outputs["answer"]
        reference = example.outputs["answer"]

        prompt = f"""
질문: {question}
정답 키워드: {reference}
생성된 답변: {answer}

생성된 답변이 질문에 관련성이 있고 정답 키워드를 포함하는지 평가하세요.
0~1 사이의 점수만 응답하세요.
"""
        response = judge.invoke(prompt)
        try:
            score = float(response.content.strip())
        except:
            score = 0.5

        return {"key": "relevance", "score": score}

    results = evaluate(
        predict,
        data="rag-chatbot-eval",
        evaluators=[relevance_evaluator],
        experiment_prefix="rag-eval"
    )

    return results


if __name__ == "__main__":
    # 평가 데이터셋 생성
    # create_evaluation_dataset()

    # 평가 실행
    results = run_evaluation()
    print(f"평가 완료: {results}")
```

---

## 10단계: 실행 가이드

### 10.1 전체 실행 순서

```bash
# 1. 프로젝트 디렉토리로 이동
cd rag-chatbot

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력

# 5. 문서 추가
# data/documents/ 디렉토리에 PDF, TXT, MD 파일 추가

# 6. 애플리케이션 실행
python -m app.main
```

### 10.2 Docker 배포

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.main"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-chatbot:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_PROJECT=rag-chatbot
    volumes:
      - ./data:/app/data
      - ./vectordb:/app/vectordb
    stdin_open: true
    tty: true
```

---

## 마무리

### 학습한 내용

1. ✅ LangChain을 사용한 문서 로딩 및 처리
2. ✅ 벡터 스토어 구축 및 의미론적 검색
3. ✅ LangGraph를 활용한 대화형 워크플로우
4. ✅ LangSmith를 통한 모니터링 및 평가

### 다음 단계

- [튜토리얼 2: 멀티 에이전트 시스템](./02-multi-agent-system.md)
- [튜토리얼 3: 대화형 AI 어시스턴트](./03-conversational-assistant.md)

### 추가 개선 아이디어

1. **하이브리드 검색**: 키워드 검색 + 벡터 검색 결합
2. **리랭킹**: 검색 결과 재정렬로 정확도 향상
3. **멀티모달**: 이미지, 표 등 다양한 문서 형식 지원
4. **캐싱**: 자주 묻는 질문에 대한 응답 캐싱
5. **피드백 루프**: 사용자 피드백을 통한 지속적 개선
