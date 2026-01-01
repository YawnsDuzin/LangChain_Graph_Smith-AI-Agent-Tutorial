# LangChain 완벽 가이드

## 목차
1. [LangChain이란?](#langchain이란)
2. [핵심 개념](#핵심-개념)
3. [아키텍처](#아키텍처)
4. [주요 컴포넌트](#주요-컴포넌트)
5. [설치 및 설정](#설치-및-설정)
6. [기본 사용법](#기본-사용법)
7. [고급 기능](#고급-기능)
8. [베스트 프랙티스](#베스트-프랙티스)

---

## LangChain이란?

LangChain은 **대규모 언어 모델(LLM)** 기반 애플리케이션을 개발하기 위한 오픈소스 프레임워크입니다. 2022년 Harrison Chase에 의해 시작되었으며, LLM을 외부 데이터 소스, 도구, 그리고 다양한 시스템과 연결하여 더욱 강력하고 유용한 AI 애플리케이션을 구축할 수 있게 해줍니다.

### 왜 LangChain을 사용해야 하는가?

1. **모듈화된 컴포넌트**: 재사용 가능한 컴포넌트들을 조합하여 복잡한 애플리케이션 구축
2. **다양한 LLM 지원**: OpenAI, Anthropic, Google, 로컬 모델 등 다양한 LLM 프로바이더 지원
3. **외부 데이터 연결**: 데이터베이스, API, 파일 시스템 등과의 원활한 통합
4. **체인 구성**: 여러 작업을 순차적 또는 병렬로 연결하여 복잡한 워크플로우 구현
5. **메모리 관리**: 대화 기록 관리 및 컨텍스트 유지
6. **활발한 커뮤니티**: 빠르게 성장하는 생태계와 풍부한 문서

---

## 핵심 개념

### 1. 체인(Chains)
체인은 LangChain의 가장 기본적인 구성 요소입니다. 여러 컴포넌트를 순차적으로 연결하여 하나의 워크플로우를 구성합니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 간단한 체인 예시
prompt = ChatPromptTemplate.from_template("다음 주제에 대해 설명해주세요: {topic}")
model = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

chain = prompt | model | output_parser
result = chain.invoke({"topic": "인공지능"})
```

### 2. 프롬프트 템플릿(Prompt Templates)
동적인 프롬프트를 생성하기 위한 템플릿 시스템입니다.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 시스템 메시지와 사용자 입력을 포함하는 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

### 3. 출력 파서(Output Parsers)
LLM의 출력을 구조화된 형식으로 변환합니다.

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Recipe(BaseModel):
    name: str = Field(description="요리 이름")
    ingredients: list[str] = Field(description="재료 목록")
    steps: list[str] = Field(description="조리 단계")

parser = JsonOutputParser(pydantic_object=Recipe)
```

### 4. 리트리버(Retrievers)
외부 데이터 소스에서 관련 정보를 검색합니다.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 벡터 스토어 기반 리트리버
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 5. 도구(Tools)
LLM이 외부 시스템과 상호작용할 수 있게 해주는 함수들입니다.

```python
from langchain_core.tools import tool

@tool
def search_weather(city: str) -> str:
    """주어진 도시의 날씨 정보를 검색합니다."""
    # 날씨 API 호출 로직
    return f"{city}의 현재 날씨는 맑음, 기온 22°C입니다."
```

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangChain 아키텍처                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Application Layer                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │   Chains    │  │   Agents    │  │   Retrievers    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Core Components                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │   │
│  │  │ Prompts  │  │ Models   │  │ Parsers  │  │ Memory  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Integration Layer                      │   │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌───────────┐  │   │
│  │  │  LLMs   │  │ Vector   │  │  APIs   │  │ Databases │  │   │
│  │  │         │  │  Stores  │  │         │  │           │  │   │
│  │  └─────────┘  └──────────┘  └─────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 주요 컴포넌트

### LangChain Core (`langchain-core`)
- 기본 추상화 및 LangChain Expression Language (LCEL)
- 프롬프트, 모델, 출력 파서의 기본 인터페이스
- 런너블(Runnable) 인터페이스

### LangChain Community (`langchain-community`)
- 서드파티 통합 (벡터 스토어, 문서 로더 등)
- 커뮤니티 기여 컴포넌트

### LangChain OpenAI (`langchain-openai`)
- OpenAI 모델 통합
- GPT-4, GPT-3.5, 임베딩 모델 지원

### LangChain 텍스트 스플리터
- 문서를 청크로 분할
- 다양한 분할 전략 지원

---

## 설치 및 설정

### 기본 설치

```bash
# 기본 패키지
pip install langchain

# OpenAI 통합
pip install langchain-openai

# 커뮤니티 패키지
pip install langchain-community

# 벡터 스토어 (Chroma)
pip install chromadb

# 전체 설치
pip install langchain langchain-openai langchain-community chromadb
```

### 환경 변수 설정

```bash
# .env 파일
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-project
```

### Python 설정

```python
import os
from dotenv import load_dotenv

load_dotenv()

# API 키 확인
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 설정되지 않았습니다."
```

---

## 기본 사용법

### 1. 간단한 대화

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 모델 초기화
chat = ChatOpenAI(model="gpt-4", temperature=0.7)

# 메시지 전송
messages = [
    SystemMessage(content="당신은 친절한 AI 어시스턴트입니다."),
    HumanMessage(content="안녕하세요! 오늘 날씨가 어때요?")
]

response = chat.invoke(messages)
print(response.content)
```

### 2. LCEL (LangChain Expression Language) 사용

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 체인 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {specialty} 전문가입니다. 간결하게 답변해주세요."),
    ("human", "{question}")
])

model = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

# LCEL로 체인 연결
chain = prompt | model | output_parser

# 실행
result = chain.invoke({
    "specialty": "Python 프로그래밍",
    "question": "리스트 컴프리헨션이란 무엇인가요?"
})
print(result)
```

### 3. 스트리밍 출력

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("다음 주제에 대해 자세히 설명해주세요: {topic}")
model = ChatOpenAI(model="gpt-4", streaming=True)

chain = prompt | model

# 스트리밍으로 출력
for chunk in chain.stream({"topic": "양자 컴퓨팅"}):
    print(chunk.content, end="", flush=True)
```

### 4. 배치 처리

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{country}의 수도는 무엇인가요?")
model = ChatOpenAI(model="gpt-4")

chain = prompt | model

# 여러 입력을 한 번에 처리
countries = [
    {"country": "한국"},
    {"country": "일본"},
    {"country": "프랑스"}
]

results = chain.batch(countries)
for result in results:
    print(result.content)
```

---

## 고급 기능

### 1. 메모리 관리

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 세션별 메모리 저장소
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 프롬프트 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

model = ChatOpenAI(model="gpt-4")
chain = prompt | model

# 메모리가 있는 체인
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 대화
config = {"configurable": {"session_id": "user-123"}}

response1 = chain_with_history.invoke(
    {"input": "제 이름은 홍길동입니다."},
    config=config
)

response2 = chain_with_history.invoke(
    {"input": "제 이름이 뭐라고 했죠?"},
    config=config
)
# "홍길동"이라고 기억하고 답변
```

### 2. 문서 로딩 및 분할

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 텍스트 파일 로딩
text_loader = TextLoader("document.txt")
text_docs = text_loader.load()

# PDF 파일 로딩
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

# 웹 페이지 로딩
web_loader = WebBaseLoader("https://example.com")
web_docs = web_loader.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_documents(pdf_docs)
```

### 3. 벡터 스토어 및 임베딩

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 임베딩 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 스토어 생성
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 유사도 검색
results = vectorstore.similarity_search(
    query="LangChain의 주요 기능은?",
    k=4
)

# 점수와 함께 검색
results_with_scores = vectorstore.similarity_search_with_score(
    query="LangChain의 주요 기능은?",
    k=4
)
```

### 4. 에이전트 구현

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 도구 정의
@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다. 예: '2 + 2' 또는 '10 * 5'"""
    try:
        return str(eval(expression))
    except:
        return "계산할 수 없습니다."

@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculate, get_current_time]

# 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도구를 사용할 수 있는 AI 어시스턴트입니다."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 에이전트 생성
model = ChatOpenAI(model="gpt-4")
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = agent_executor.invoke({
    "input": "현재 시간을 알려주고, 1234 * 5678을 계산해주세요."
})
```

---

## 베스트 프랙티스

### 1. 프롬프트 엔지니어링
- 명확하고 구체적인 지시사항 작성
- 예시를 포함하여 원하는 출력 형식 명시
- 시스템 프롬프트로 AI의 역할과 제약 조건 설정

### 2. 에러 처리
```python
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")

try:
    result = model.invoke(
        "질문",
        config=RunnableConfig(
            max_retries=3,
            timeout=30
        )
    )
except Exception as e:
    print(f"에러 발생: {e}")
    # 폴백 로직 실행
```

### 3. 비용 최적화
- 적절한 모델 선택 (간단한 작업은 GPT-3.5, 복잡한 작업은 GPT-4)
- 캐싱 활용
- 토큰 사용량 모니터링

### 4. 성능 최적화
```python
from langchain_core.runnables import RunnableParallel

# 병렬 실행
parallel_chain = RunnableParallel(
    summary=summarize_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

# 모든 작업이 병렬로 실행됨
results = parallel_chain.invoke({"text": "분석할 텍스트"})
```

### 5. 테스트
```python
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 테스트
prompt = ChatPromptTemplate.from_template("안녕하세요, {name}님!")
formatted = prompt.format(name="홍길동")
assert "홍길동" in formatted
```

---

## 다음 단계

- [LangGraph 가이드](./02-langgraph-overview.md): 복잡한 에이전트 워크플로우 구축
- [LangSmith 가이드](./03-langsmith-overview.md): 모니터링 및 디버깅
- [튜토리얼: RAG 챗봇 구축](../tutorials/01-rag-chatbot.md)
- [튜토리얼: 멀티 에이전트 시스템](../tutorials/02-multi-agent-system.md)
- [튜토리얼: 대화형 AI 어시스턴트](../tutorials/03-conversational-assistant.md)
