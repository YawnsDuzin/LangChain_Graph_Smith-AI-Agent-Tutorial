# LangChain, LangGraph, LangSmith AI 에이전트 튜토리얼

LangChain, LangGraph, LangSmith를 활용한 AI 에이전트 구축 가이드 및 실습 예제 모음입니다.

## 개요

이 저장소는 LLM 기반 AI 에이전트 개발에 필요한 세 가지 핵심 오픈소스 도구에 대한 종합 가이드를 제공합니다:

| 도구 | 설명 | 주요 용도 |
|------|------|----------|
| **LangChain** | LLM 애플리케이션 개발 프레임워크 | 체인, 도구, 메모리 관리 |
| **LangGraph** | 그래프 기반 워크플로우 라이브러리 | 멀티 에이전트, 상태 관리 |
| **LangSmith** | LLM 모니터링 및 평가 플랫폼 | 트레이싱, 디버깅, 평가 |

## 프로젝트 구조

```
LangChain_Graph_Smith-AI-Agent-Tutorial/
├── docs/
│   ├── overview/                    # 개념 설명 문서
│   │   ├── 01-langchain-overview.md  # LangChain 완벽 가이드
│   │   ├── 02-langgraph-overview.md  # LangGraph 완벽 가이드
│   │   └── 03-langsmith-overview.md  # LangSmith 완벽 가이드
│   │
│   └── tutorials/                   # 실습 튜토리얼
│       ├── 01-rag-chatbot.md         # RAG 챗봇 구축
│       ├── 02-multi-agent-system.md  # 멀티 에이전트 시스템
│       └── 03-conversational-assistant.md # 대화형 AI 어시스턴트
│
├── examples/                        # 실행 가능한 예제 코드
│   ├── 01-rag-chatbot/               # RAG 챗봇 예제
│   ├── 02-multi-agent-system/        # 멀티 에이전트 예제
│   └── 03-conversational-assistant/  # 대화형 어시스턴트 예제
│
└── src/                             # 공통 유틸리티
    ├── utils/
    └── config/
```

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/YawnsDuzin/LangChain_Graph_Smith-AI-Agent-Tutorial.git
cd LangChain_Graph_Smith-AI-Agent-Tutorial
```

### 2. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 공통 의존성 설치
pip install langchain langchain-openai langchain-community langgraph langsmith python-dotenv
```

### 3. API 키 설정

`.env` 파일을 생성하고 API 키를 추가합니다:

```bash
# .env
OPENAI_API_KEY=sk-your-openai-api-key
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-project

# 선택적
TAVILY_API_KEY=tvly-your-tavily-key
```

### 4. 예제 실행

```bash
# RAG 챗봇 예제
cd examples/01-rag-chatbot
pip install -r requirements.txt
python main.py

# 멀티 에이전트 시스템 예제
cd examples/02-multi-agent-system
pip install -r requirements.txt
python main.py

# 대화형 어시스턴트 예제
cd examples/03-conversational-assistant
pip install -r requirements.txt
python main.py
```

## 문서 가이드

### 📚 개념 문서

1. **[LangChain 완벽 가이드](docs/overview/01-langchain-overview.md)**
   - LangChain 핵심 개념
   - 체인, 프롬프트, 도구 사용법
   - LCEL (LangChain Expression Language)
   - 메모리 관리

2. **[LangGraph 완벽 가이드](docs/overview/02-langgraph-overview.md)**
   - 그래프 기반 워크플로우
   - 상태 관리와 체크포인팅
   - 조건부 라우팅
   - 멀티 에이전트 패턴

3. **[LangSmith 완벽 가이드](docs/overview/03-langsmith-overview.md)**
   - 트레이싱 설정
   - 평가(Evaluation) 시스템
   - 프롬프트 관리
   - 모니터링 및 알림

### 🛠 실습 튜토리얼

1. **[RAG 챗봇 구축](docs/tutorials/01-rag-chatbot.md)**
   - 문서 로딩 및 처리
   - 벡터 스토어 구축
   - 대화형 RAG 체인
   - LangSmith 모니터링

2. **[멀티 에이전트 시스템](docs/tutorials/02-multi-agent-system.md)**
   - 에이전트 역할 설계
   - 슈퍼바이저 패턴
   - 에이전트 간 협업
   - 동적 에이전트 생성

3. **[대화형 AI 어시스턴트](docs/tutorials/03-conversational-assistant.md)**
   - 도구 사용 에이전트
   - Human-in-the-Loop
   - 스트리밍 응답
   - 웹 인터페이스

## 예제 코드

| 예제 | 설명 | 핵심 기술 |
|------|------|----------|
| [RAG 챗봇](examples/01-rag-chatbot/) | 문서 기반 Q&A 시스템 | LangChain, ChromaDB |
| [멀티 에이전트](examples/02-multi-agent-system/) | 협업하는 AI 팀 | LangGraph, 슈퍼바이저 패턴 |
| [대화형 어시스턴트](examples/03-conversational-assistant/) | 도구 사용 챗봇 | 도구 통합, 메모리 |

## 아키텍처

### LangChain + LangGraph + LangSmith 통합

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI 에이전트 시스템                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   사용자 입력                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LangGraph                            │   │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │   │
│   │  │ 노드 1  │───▶│ 노드 2  │───▶│ 노드 3  │             │   │
│   │  │(분석)   │    │(처리)   │    │(응답)   │             │   │
│   │  └─────────┘    └─────────┘    └─────────┘             │   │
│   │       │              │              │                   │   │
│   └───────┼──────────────┼──────────────┼───────────────────┘   │
│           │              │              │                       │
│   ┌───────┼──────────────┼──────────────┼───────────────────┐   │
│   │       ▼              ▼              ▼      LangChain    │   │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │   │
│   │  │ 프롬프트 │    │  LLM   │    │  도구   │             │   │
│   │  │ 템플릿  │    │        │    │        │             │   │
│   │  └─────────┘    └─────────┘    └─────────┘             │   │
│   └─────────────────────────────────────────────────────────┘   │
│           │              │              │                       │
│   ┌───────┴──────────────┴──────────────┴───────────────────┐   │
│   │                    LangSmith                            │   │
│   │  📊 트레이싱    📈 평가    🔔 모니터링    📝 프롬프트 관리  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 주요 기능

### LangChain
- ✅ 다양한 LLM 프로바이더 지원 (OpenAI, Anthropic, Google 등)
- ✅ 체인 구성을 위한 LCEL
- ✅ 문서 로딩 및 텍스트 분할
- ✅ 벡터 스토어 통합
- ✅ 도구 및 에이전트

### LangGraph
- ✅ 상태 기반 그래프 워크플로우
- ✅ 순환 그래프 지원
- ✅ 체크포인팅 및 상태 복원
- ✅ Human-in-the-Loop
- ✅ 스트리밍 지원

### LangSmith
- ✅ 자동 트레이싱
- ✅ 자동화된 평가
- ✅ 프롬프트 버전 관리
- ✅ 데이터셋 관리
- ✅ 실시간 모니터링

## 요구 사항

- Python 3.9+
- OpenAI API 키
- LangSmith API 키 (모니터링용, 선택사항)
- Tavily API 키 (웹 검색용, 선택사항)

## 라이센스

MIT License

## 기여하기

이슈와 풀 리퀘스트를 환영합니다!

## 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangSmith 공식 문서](https://docs.smith.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
