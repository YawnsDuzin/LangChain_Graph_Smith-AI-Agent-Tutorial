# 튜토리얼 2: 멀티 에이전트 협업 시스템 구축

## 개요

이 튜토리얼에서는 **LangGraph**를 활용하여 여러 AI 에이전트가 협업하는 시스템을 구축합니다. 연구, 작성, 검토 등 각기 다른 역할을 가진 에이전트들이 슈퍼바이저의 조율 하에 복잡한 작업을 수행하는 방법을 배웁니다.

### 학습 목표

- 멀티 에이전트 아키텍처 설계
- LangGraph를 활용한 에이전트 간 협업 구현
- 도구 사용 에이전트 구현
- 슈퍼바이저 패턴 적용
- LangSmith를 통한 에이전트 행동 추적

### 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│              멀티 에이전트 협업 시스템 아키텍처                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌──────────────┐                           │
│                      │   사용자     │                           │
│                      └──────┬───────┘                           │
│                             │                                   │
│                             ▼                                   │
│                     ┌───────────────┐                           │
│             ┌───────│  슈퍼바이저    │◄──────────┐               │
│             │       │  (Supervisor) │           │               │
│             │       └───────┬───────┘           │               │
│             │               │                   │               │
│             │    ┌──────────┼──────────┐       │               │
│             │    │          │          │       │               │
│             ▼    ▼          ▼          ▼       │               │
│         ┌────────┐    ┌────────┐    ┌────────┐ │               │
│         │연구원   │    │작성자   │    │검토자   │ │               │
│         │Agent   │    │Agent   │    │Agent   │ │               │
│         └───┬────┘    └───┬────┘    └───┬────┘ │               │
│             │             │             │       │               │
│             ▼             ▼             ▼       │               │
│         ┌────────┐    ┌────────┐    ┌────────┐ │               │
│         │🔍 검색 │    │📝 작성 │    │✅ 검토 │ │               │
│         │  도구  │    │  도구  │    │  도구  │ │               │
│         └────────┘    └────────┘    └────────┘ │               │
│             │             │             │       │               │
│             └─────────────┴─────────────┴───────┘               │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │ 최종 결과물  │                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1단계: 프로젝트 설정

### 1.1 디렉토리 구조

```
multi-agent-system/
├── agents/
│   ├── __init__.py
│   ├── base.py            # 기본 에이전트 클래스
│   ├── researcher.py      # 연구 에이전트
│   ├── writer.py          # 작성 에이전트
│   ├── reviewer.py        # 검토 에이전트
│   └── supervisor.py      # 슈퍼바이저
├── tools/
│   ├── __init__.py
│   ├── search.py          # 검색 도구
│   ├── write.py           # 작성 도구
│   └── review.py          # 검토 도구
├── graph/
│   ├── __init__.py
│   ├── state.py           # 상태 정의
│   └── workflow.py        # 그래프 워크플로우
├── app/
│   ├── __init__.py
│   ├── config.py
│   └── main.py
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
tavily-python>=0.3.0
python-dotenv>=1.0.0
```

```bash
pip install -r requirements.txt
```

### 1.3 환경 변수

```bash
# .env
OPENAI_API_KEY=sk-your-openai-api-key
TAVILY_API_KEY=tvly-your-tavily-key
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multi-agent-tutorial
```

---

## 2단계: 상태 정의

### 2.1 graph/state.py

```python
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
```

---

## 3단계: 도구 구현

### 3.1 tools/search.py

```python
# tools/search.py
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List, Dict, Any
import os


@tool
def web_search(query: str) -> str:
    """
    웹에서 정보를 검색합니다.

    Args:
        query: 검색할 쿼리

    Returns:
        검색 결과 요약
    """
    try:
        search = TavilySearchResults(max_results=5)
        results = search.invoke(query)

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"[{i}] {result.get('title', 'No Title')}\n"
                f"    URL: {result.get('url', 'N/A')}\n"
                f"    내용: {result.get('content', 'No content')[:200]}..."
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"


@tool
def academic_search(query: str, num_results: int = 3) -> str:
    """
    학술 자료를 검색합니다.

    Args:
        query: 검색할 학술 주제
        num_results: 반환할 결과 수

    Returns:
        학술 검색 결과
    """
    # 실제로는 학술 API (예: Semantic Scholar, arXiv)를 사용
    # 여기서는 시뮬레이션
    return f"""
학술 검색 결과 ('{query}'):

[1] "Recent Advances in {query}" (2024)
    저자: Kim et al.
    요약: 최신 연구 동향과 주요 발견을 다룹니다.

[2] "A Comprehensive Survey on {query}" (2023)
    저자: Lee et al.
    요약: 해당 분야의 종합적인 서베이 논문입니다.

[3] "Practical Applications of {query}" (2024)
    저자: Park et al.
    요약: 실제 응용 사례와 구현 방법을 설명합니다.
"""


@tool
def analyze_sources(sources: List[str]) -> Dict[str, Any]:
    """
    수집된 자료를 분석합니다.

    Args:
        sources: 분석할 자료 목록

    Returns:
        분석 결과
    """
    return {
        "total_sources": len(sources),
        "reliability_score": 0.85,
        "key_themes": ["주제1", "주제2", "주제3"],
        "consensus": "대부분의 자료가 일관된 관점을 제시합니다."
    }


# 연구 도구 모음
research_tools = [web_search, academic_search, analyze_sources]
```

### 3.2 tools/write.py

```python
# tools/write.py
from langchain_core.tools import tool
from typing import List, Dict


@tool
def create_outline(topic: str, sections: int = 5) -> Dict:
    """
    글의 개요를 생성합니다.

    Args:
        topic: 글의 주제
        sections: 섹션 수

    Returns:
        글 개요
    """
    return {
        "title": f"{topic}에 대한 종합 가이드",
        "sections": [
            {"number": 1, "title": "소개", "description": "주제 소개 및 배경"},
            {"number": 2, "title": "핵심 개념", "description": "주요 개념 설명"},
            {"number": 3, "title": "상세 분석", "description": "깊이 있는 분석"},
            {"number": 4, "title": "실제 적용", "description": "적용 사례"},
            {"number": 5, "title": "결론", "description": "요약 및 향후 전망"}
        ]
    }


@tool
def write_section(section_title: str, content_points: List[str]) -> str:
    """
    특정 섹션의 내용을 작성합니다.

    Args:
        section_title: 섹션 제목
        content_points: 포함할 핵심 포인트

    Returns:
        작성된 섹션 내용
    """
    points_text = "\n".join([f"- {point}" for point in content_points])
    return f"""
## {section_title}

{points_text}

이 섹션에서는 위의 핵심 포인트들을 상세히 다루었습니다.
각 포인트는 연구 자료를 바탕으로 작성되었습니다.
"""


@tool
def format_document(
    title: str,
    sections: List[Dict],
    format_type: str = "markdown"
) -> str:
    """
    문서를 최종 형식으로 포맷팅합니다.

    Args:
        title: 문서 제목
        sections: 섹션 목록
        format_type: 출력 형식 (markdown, html, plain)

    Returns:
        포맷팅된 문서
    """
    if format_type == "markdown":
        doc = f"# {title}\n\n"
        for section in sections:
            doc += f"## {section.get('title', 'Untitled')}\n\n"
            doc += f"{section.get('content', '')}\n\n"
        return doc
    return str(sections)


@tool
def calculate_word_count(text: str) -> Dict:
    """
    텍스트의 단어 수와 통계를 계산합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        텍스트 통계
    """
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')

    return {
        "word_count": len(words),
        "sentence_count": sentences,
        "avg_words_per_sentence": len(words) / max(sentences, 1),
        "character_count": len(text)
    }


# 작성 도구 모음
writing_tools = [create_outline, write_section, format_document, calculate_word_count]
```

### 3.3 tools/review.py

```python
# tools/review.py
from langchain_core.tools import tool
from typing import Dict, List


@tool
def check_grammar(text: str) -> Dict:
    """
    문법과 맞춤법을 검사합니다.

    Args:
        text: 검사할 텍스트

    Returns:
        검사 결과
    """
    # 실제로는 문법 검사 API 사용
    return {
        "errors_found": 0,
        "suggestions": [],
        "readability_score": 85,
        "grade_level": "대학교 수준"
    }


@tool
def check_factual_accuracy(
    claims: List[str],
    sources: List[str]
) -> Dict:
    """
    주장의 사실 정확성을 검증합니다.

    Args:
        claims: 검증할 주장 목록
        sources: 참조 자료

    Returns:
        검증 결과
    """
    return {
        "verified_claims": len(claims),
        "accuracy_score": 0.92,
        "unverified_claims": [],
        "sources_cited": len(sources)
    }


@tool
def evaluate_structure(document: str) -> Dict:
    """
    문서 구조를 평가합니다.

    Args:
        document: 평가할 문서

    Returns:
        구조 평가 결과
    """
    sections = document.count("##")
    paragraphs = document.count("\n\n")

    return {
        "section_count": sections,
        "paragraph_count": paragraphs,
        "structure_score": 0.88,
        "suggestions": [
            "각 섹션에 더 많은 예시 추가 권장",
            "결론 섹션 보강 필요"
        ]
    }


@tool
def generate_feedback(
    content: str,
    criteria: List[str]
) -> Dict:
    """
    종합적인 피드백을 생성합니다.

    Args:
        content: 검토할 콘텐츠
        criteria: 평가 기준

    Returns:
        종합 피드백
    """
    return {
        "overall_score": 0.85,
        "strengths": [
            "명확한 구조",
            "풍부한 예시",
            "논리적 흐름"
        ],
        "areas_for_improvement": [
            "일부 전문 용어 설명 필요",
            "참고 문헌 추가 권장"
        ],
        "recommendation": "수정 후 게시 권장",
        "approved": True
    }


# 검토 도구 모음
review_tools = [check_grammar, check_factual_accuracy, evaluate_structure, generate_feedback]
```

---

## 4단계: 에이전트 구현

### 4.1 agents/base.py

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langsmith import traceable
import os


class BaseAgent(ABC):
    """기본 에이전트 클래스"""

    def __init__(
        self,
        name: str,
        role: str,
        tools: List = None,
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self.name = name
        self.role = role
        self.tools = tools or []
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """에이전트의 시스템 프롬프트"""
        pass

    def create_prompt(self) -> ChatPromptTemplate:
        """프롬프트 템플릿 생성"""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ])

    @traceable(name="agent_invoke")
    def invoke(self, state: Dict) -> Dict:
        """에이전트 실행"""
        messages = state.get("messages", [])
        task = state.get("task", "")

        prompt = self.create_prompt()
        chain = prompt | self.llm

        response = chain.invoke({
            "messages": messages,
            "input": task
        })

        return self.process_response(response, state)

    @abstractmethod
    def process_response(self, response: Any, state: Dict) -> Dict:
        """응답 처리"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, role={self.role})"
```

### 4.2 agents/researcher.py

```python
# agents/researcher.py
from typing import Dict, Any
from langchain_core.messages import AIMessage
from agents.base import BaseAgent
from tools.search import research_tools
from langsmith import traceable


class ResearcherAgent(BaseAgent):
    """연구 전문 에이전트"""

    def __init__(self):
        super().__init__(
            name="Researcher",
            role="연구 전문가",
            tools=research_tools,
            temperature=0.3  # 정확성을 위해 낮은 temperature
        )

    @property
    def system_prompt(self) -> str:
        return """당신은 철저하고 정확한 연구 전문가입니다.

역할:
- 주어진 주제에 대해 깊이 있는 연구를 수행합니다.
- 신뢰할 수 있는 자료를 찾고 분석합니다.
- 핵심 발견사항을 정리합니다.

규칙:
1. 항상 여러 출처를 확인하세요.
2. 사실과 의견을 구분하세요.
3. 최신 정보를 우선시하세요.
4. 불확실한 정보는 명시하세요.

사용 가능한 도구:
- web_search: 웹에서 정보 검색
- academic_search: 학술 자료 검색
- analyze_sources: 자료 분석

연구 결과는 다음 형식으로 제공하세요:
- 주요 발견사항
- 출처 목록
- 신뢰도 평가"""

    @traceable(name="researcher_process")
    def process_response(self, response: Any, state: Dict) -> Dict:
        """연구 결과 처리"""
        content = response.content if hasattr(response, 'content') else str(response)

        # 도구 호출 처리
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = self._execute_tools(response.tool_calls)
            content += f"\n\n도구 실행 결과:\n{tool_results}"

        research_data = {
            "topic": state.get("task", ""),
            "sources": self._extract_sources(content),
            "findings": content,
            "confidence": 0.85  # 신뢰도 점수
        }

        return {
            "messages": [AIMessage(content=content, name=self.name)],
            "research_data": research_data,
            "next_agent": "supervisor"
        }

    def _execute_tools(self, tool_calls) -> str:
        """도구 실행"""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            for tool in self.tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    results.append(f"{tool_name}: {result}")
                    break

        return "\n".join(results)

    def _extract_sources(self, content: str) -> list:
        """출처 추출"""
        # 간단한 URL 추출 로직
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', content)
        return urls if urls else ["내부 분석 기반"]
```

### 4.3 agents/writer.py

```python
# agents/writer.py
from typing import Dict, Any
from langchain_core.messages import AIMessage
from agents.base import BaseAgent
from tools.write import writing_tools
from langsmith import traceable


class WriterAgent(BaseAgent):
    """콘텐츠 작성 에이전트"""

    def __init__(self):
        super().__init__(
            name="Writer",
            role="콘텐츠 작성자",
            tools=writing_tools,
            temperature=0.7  # 창의성을 위해 적당한 temperature
        )

    @property
    def system_prompt(self) -> str:
        return """당신은 전문적인 콘텐츠 작성자입니다.

역할:
- 연구 자료를 바탕으로 고품질 콘텐츠를 작성합니다.
- 명확하고 읽기 쉬운 글을 작성합니다.
- 적절한 구조와 흐름을 유지합니다.

규칙:
1. 연구 결과를 정확하게 반영하세요.
2. 독자 수준에 맞는 언어를 사용하세요.
3. 논리적인 구조를 유지하세요.
4. 적절한 예시와 설명을 포함하세요.

사용 가능한 도구:
- create_outline: 개요 생성
- write_section: 섹션 작성
- format_document: 문서 포맷팅
- calculate_word_count: 단어 수 계산

작성 시 다음을 포함하세요:
- 명확한 제목
- 잘 구성된 섹션
- 핵심 메시지"""

    @traceable(name="writer_process")
    def process_response(self, response: Any, state: Dict) -> Dict:
        """작성 결과 처리"""
        content = response.content if hasattr(response, 'content') else str(response)

        # 연구 데이터 참조
        research_data = state.get("research_data", {})
        if research_data:
            content = self._incorporate_research(content, research_data)

        draft_content = {
            "title": self._extract_title(content),
            "sections": self._extract_sections(content),
            "word_count": len(content.split())
        }

        return {
            "messages": [AIMessage(content=content, name=self.name)],
            "draft_content": draft_content,
            "next_agent": "supervisor"
        }

    def _incorporate_research(self, content: str, research_data: Dict) -> str:
        """연구 데이터 통합"""
        findings = research_data.get("findings", "")
        if findings:
            content = f"[연구 기반 작성]\n\n{content}\n\n참고: {findings[:500]}..."
        return content

    def _extract_title(self, content: str) -> str:
        """제목 추출"""
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                return line.strip().lstrip('#').strip()
        return "제목 없음"

    def _extract_sections(self, content: str) -> list:
        """섹션 추출"""
        sections = []
        current_section = {"title": "", "content": ""}

        for line in content.split('\n'):
            if line.startswith('##'):
                if current_section["title"]:
                    sections.append(current_section)
                current_section = {
                    "title": line.lstrip('#').strip(),
                    "content": ""
                }
            else:
                current_section["content"] += line + "\n"

        if current_section["title"]:
            sections.append(current_section)

        return sections
```

### 4.4 agents/reviewer.py

```python
# agents/reviewer.py
from typing import Dict, Any
from langchain_core.messages import AIMessage
from agents.base import BaseAgent
from tools.review import review_tools
from langsmith import traceable


class ReviewerAgent(BaseAgent):
    """콘텐츠 검토 에이전트"""

    def __init__(self):
        super().__init__(
            name="Reviewer",
            role="콘텐츠 검토자",
            tools=review_tools,
            temperature=0.3  # 일관된 평가를 위해 낮은 temperature
        )

    @property
    def system_prompt(self) -> str:
        return """당신은 꼼꼼하고 공정한 콘텐츠 검토자입니다.

역할:
- 작성된 콘텐츠의 품질을 평가합니다.
- 건설적인 피드백을 제공합니다.
- 개선 사항을 명확히 제시합니다.

평가 기준:
1. 정확성: 정보가 정확한가?
2. 명확성: 이해하기 쉬운가?
3. 구조: 논리적으로 구성되었는가?
4. 완성도: 필요한 내용이 모두 포함되었는가?

사용 가능한 도구:
- check_grammar: 문법 검사
- check_factual_accuracy: 사실 검증
- evaluate_structure: 구조 평가
- generate_feedback: 피드백 생성

검토 결과는 다음을 포함해야 합니다:
- 점수 (0-100)
- 강점
- 개선 필요 사항
- 승인 여부"""

    @traceable(name="reviewer_process")
    def process_response(self, response: Any, state: Dict) -> Dict:
        """검토 결과 처리"""
        content = response.content if hasattr(response, 'content') else str(response)

        # 초안 분석
        draft = state.get("draft_content", {})

        # 피드백 생성
        feedback = self._generate_feedback(content, draft)

        return {
            "messages": [AIMessage(content=content, name=self.name)],
            "review_feedback": feedback,
            "next_agent": "supervisor"
        }

    def _generate_feedback(self, review_content: str, draft: Dict) -> Dict:
        """피드백 구조화"""
        # 점수 추출 시도
        score = self._extract_score(review_content)

        return {
            "score": score,
            "strengths": self._extract_list(review_content, "강점"),
            "improvements": self._extract_list(review_content, "개선"),
            "approved": score >= 70,
            "raw_feedback": review_content
        }

    def _extract_score(self, content: str) -> float:
        """점수 추출"""
        import re
        # 숫자 패턴 찾기 (예: 85점, 85/100, 85%)
        patterns = [
            r'(\d+)\s*점',
            r'(\d+)\s*/\s*100',
            r'(\d+)\s*%',
            r'점수[:\s]*(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return float(match.group(1))
        return 75.0  # 기본값

    def _extract_list(self, content: str, keyword: str) -> list:
        """목록 추출"""
        items = []
        lines = content.split('\n')
        in_section = False

        for line in lines:
            if keyword in line:
                in_section = True
                continue
            if in_section:
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    items.append(line.strip().lstrip('-•').strip())
                elif line.strip() and not line.strip().startswith('#'):
                    if len(items) >= 3:  # 최대 3개
                        break

        return items if items else ["평가 진행 중"]
```

### 4.5 agents/supervisor.py

```python
# agents/supervisor.py
from typing import Dict, Any, Literal
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable


class SupervisorAgent:
    """워크플로우를 조율하는 슈퍼바이저 에이전트"""

    AGENTS = ["researcher", "writer", "reviewer", "FINISH"]

    def __init__(self, model: str = "gpt-4", temperature: float = 0):
        self.name = "Supervisor"
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    @property
    def routing_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
당신은 콘텐츠 제작 팀의 슈퍼바이저입니다.
팀원들의 작업을 조율하고 다음에 어떤 에이전트가 작업해야 할지 결정합니다.

팀원:
- researcher: 정보 조사 및 연구
- writer: 콘텐츠 작성
- reviewer: 콘텐츠 검토

현재 상태:
- 작업: {task}
- 연구 완료: {research_done}
- 작성 완료: {writing_done}
- 검토 완료: {review_done}
- 검토 승인: {approved}
- 반복 횟수: {iteration}/{max_iteration}

대화 기록:
{messages}

워크플로우 규칙:
1. 먼저 researcher가 연구를 수행해야 합니다.
2. 연구가 완료되면 writer가 글을 작성합니다.
3. 글이 작성되면 reviewer가 검토합니다.
4. 검토가 승인되면 FINISH입니다.
5. 검토가 거부되면 writer가 수정합니다.
6. 최대 반복 횟수에 도달하면 FINISH입니다.

다음에 작업할 에이전트를 선택하세요.
오직 다음 중 하나만 응답하세요: {agents}

선택:""")

    @traceable(name="supervisor_route")
    def route(self, state: Dict) -> str:
        """다음 에이전트 결정"""
        # 상태 분석
        research_data = state.get("research_data")
        draft_content = state.get("draft_content")
        review_feedback = state.get("review_feedback")
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 5)

        # 메시지 포맷팅
        messages = state.get("messages", [])
        messages_str = "\n".join([
            f"{m.name if hasattr(m, 'name') else 'User'}: {m.content[:200]}..."
            for m in messages[-5:]  # 최근 5개만
        ])

        # 프롬프트 실행
        chain = self.routing_prompt | self.llm

        response = chain.invoke({
            "task": state.get("task", ""),
            "research_done": "완료" if research_data else "미완료",
            "writing_done": "완료" if draft_content else "미완료",
            "review_done": "완료" if review_feedback else "미완료",
            "approved": review_feedback.get("approved", False) if review_feedback else "N/A",
            "iteration": iteration,
            "max_iteration": max_iter,
            "messages": messages_str,
            "agents": ", ".join(self.AGENTS)
        })

        next_agent = response.content.strip().lower()

        # 유효성 검사
        if next_agent not in [a.lower() for a in self.AGENTS]:
            next_agent = self._fallback_routing(state)

        return next_agent

    def _fallback_routing(self, state: Dict) -> str:
        """폴백 라우팅 로직"""
        research_data = state.get("research_data")
        draft_content = state.get("draft_content")
        review_feedback = state.get("review_feedback")
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 5)

        # 최대 반복 도달
        if iteration >= max_iter:
            return "finish"

        # 연구 필요
        if not research_data:
            return "researcher"

        # 작성 필요
        if not draft_content:
            return "writer"

        # 검토 필요
        if not review_feedback:
            return "reviewer"

        # 검토 통과
        if review_feedback.get("approved", False):
            return "finish"

        # 수정 필요
        return "writer"

    @traceable(name="supervisor_invoke")
    def invoke(self, state: Dict) -> Dict:
        """슈퍼바이저 실행"""
        next_agent = self.route(state)

        message = f"[슈퍼바이저] 다음 작업자: {next_agent}"

        return {
            "messages": [AIMessage(content=message, name=self.name)],
            "next_agent": next_agent,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
```

---

## 5단계: 그래프 워크플로우 구현

### 5.1 graph/workflow.py

```python
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
```

---

## 6단계: 메인 애플리케이션

### 6.1 app/main.py

```python
# app/main.py
import os
from dotenv import load_dotenv
from graph.workflow import MultiAgentWorkflow
from langsmith import Client

load_dotenv()


def print_header():
    print("\n" + "="*60)
    print("🤖 멀티 에이전트 콘텐츠 제작 시스템")
    print("="*60)
    print("\n에이전트 팀:")
    print("  📚 Researcher - 정보 조사 및 연구")
    print("  ✍️  Writer    - 콘텐츠 작성")
    print("  ✅ Reviewer   - 콘텐츠 검토")
    print("  👔 Supervisor - 워크플로우 조율")
    print("-"*60)


def run_interactive():
    """대화형 모드 실행"""
    print_header()

    workflow = MultiAgentWorkflow()
    workflow.compile()

    print("\n작업을 입력하세요. 종료하려면 'quit'를 입력하세요.\n")

    while True:
        try:
            task = input("📝 작업: ").strip()

            if not task:
                continue

            if task.lower() in ['quit', 'exit', '종료']:
                print("\n👋 시스템을 종료합니다.")
                break

            print("\n🔄 멀티 에이전트가 작업을 시작합니다...\n")

            # 스트리밍 실행
            for event in workflow.stream(task):
                for node, update in event.items():
                    if "messages" in update:
                        for msg in update["messages"]:
                            name = getattr(msg, 'name', 'System')
                            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                            print(f"[{name}] {content}")
                    print()

            print("\n" + "="*60)
            print("✅ 작업이 완료되었습니다.")
            print("="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


def run_single_task(task: str):
    """단일 작업 실행"""
    print_header()

    workflow = MultiAgentWorkflow()
    result = workflow.run(task)

    print("\n" + "="*60)
    print("📊 실행 결과")
    print("="*60)

    print(f"\n상태: {result['status']}")
    print(f"총 반복 횟수: {result['iterations']}")

    if result['research_data']:
        print("\n📚 연구 결과:")
        print(f"  - 주제: {result['research_data'].get('topic', 'N/A')}")
        print(f"  - 신뢰도: {result['research_data'].get('confidence', 0):.0%}")

    if result['draft_content']:
        print("\n✍️  작성 결과:")
        print(f"  - 제목: {result['draft_content'].get('title', 'N/A')}")
        print(f"  - 단어 수: {result['draft_content'].get('word_count', 0)}")
        print(f"  - 섹션 수: {len(result['draft_content'].get('sections', []))}")

    if result['review_feedback']:
        print("\n✅ 검토 결과:")
        print(f"  - 점수: {result['review_feedback'].get('score', 0)}/100")
        print(f"  - 승인: {'예' if result['review_feedback'].get('approved') else '아니오'}")

        strengths = result['review_feedback'].get('strengths', [])
        if strengths:
            print("  - 강점:")
            for s in strengths[:3]:
                print(f"    • {s}")

    print("\n" + "="*60)


def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1:
        # 커맨드라인 인자로 작업 전달
        task = " ".join(sys.argv[1:])
        run_single_task(task)
    else:
        # 대화형 모드
        run_interactive()


if __name__ == "__main__":
    main()
```

---

## 7단계: 고급 패턴

### 7.1 계층적 에이전트 구조

```python
# agents/hierarchical.py
from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage


class TeamLeader:
    """팀 리더 에이전트 - 하위 팀을 관리"""

    def __init__(self, name: str, team_members: List):
        self.name = name
        self.team_members = team_members
        self.sub_graph = self._build_team_graph()

    def _build_team_graph(self) -> StateGraph:
        """팀 내부 그래프 구축"""
        # 팀 멤버들로 서브그래프 구성
        pass

    def delegate_task(self, task: str, state: Dict) -> Dict:
        """작업을 팀에게 위임"""
        # 적절한 팀 멤버에게 작업 할당
        pass


class HierarchicalMultiAgentSystem:
    """계층적 멀티 에이전트 시스템"""

    def __init__(self):
        # 연구팀
        self.research_team = TeamLeader(
            "Research Team Lead",
            ["data_analyst", "fact_checker", "summarizer"]
        )

        # 콘텐츠팀
        self.content_team = TeamLeader(
            "Content Team Lead",
            ["outline_creator", "section_writer", "editor"]
        )

        # QA팀
        self.qa_team = TeamLeader(
            "QA Team Lead",
            ["grammar_checker", "fact_verifier", "style_reviewer"]
        )

    def build_organization_graph(self) -> StateGraph:
        """조직 그래프 구축"""
        graph = StateGraph(AgentState)

        # 각 팀을 노드로 추가
        graph.add_node("research_team", self.research_team.delegate_task)
        graph.add_node("content_team", self.content_team.delegate_task)
        graph.add_node("qa_team", self.qa_team.delegate_task)

        # CEO/슈퍼바이저 추가
        graph.add_node("ceo", self._ceo_decision)

        # 연결
        graph.add_edge(START, "ceo")
        graph.add_conditional_edges("ceo", self._route_to_team)

        return graph

    def _ceo_decision(self, state: Dict) -> Dict:
        """CEO 의사결정"""
        pass

    def _route_to_team(self, state: Dict) -> str:
        """팀 라우팅"""
        pass
```

### 7.2 동적 에이전트 생성

```python
# agents/dynamic.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI


class DynamicAgentFactory:
    """동적으로 에이전트를 생성하는 팩토리"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.agent_registry = {}

    def create_specialist(
        self,
        specialty: str,
        tools: list = None
    ) -> "DynamicAgent":
        """전문 에이전트 생성"""
        agent_id = f"specialist_{specialty}_{len(self.agent_registry)}"

        # 시스템 프롬프트 동적 생성
        system_prompt = self._generate_system_prompt(specialty)

        agent = DynamicAgent(
            agent_id=agent_id,
            specialty=specialty,
            system_prompt=system_prompt,
            tools=tools or []
        )

        self.agent_registry[agent_id] = agent
        return agent

    def _generate_system_prompt(self, specialty: str) -> str:
        """전문 분야에 맞는 시스템 프롬프트 생성"""
        prompt = f"""당신은 {specialty} 분야의 전문가입니다.

당신의 전문 지식을 활용하여 주어진 작업을 수행하세요.
명확하고 전문적인 답변을 제공하세요.
"""
        return prompt

    def spawn_team(
        self,
        task_description: str,
        max_agents: int = 5
    ) -> List["DynamicAgent"]:
        """작업에 필요한 팀 자동 생성"""
        # LLM에게 필요한 전문가 목록 요청
        response = self.llm.invoke(
            f"다음 작업을 수행하기 위해 필요한 전문가 역할 {max_agents}개를 나열하세요: {task_description}"
        )

        specialties = self._parse_specialties(response.content)

        team = []
        for specialty in specialties:
            agent = self.create_specialist(specialty)
            team.append(agent)

        return team

    def _parse_specialties(self, response: str) -> List[str]:
        """응답에서 전문 분야 추출"""
        lines = response.strip().split('\n')
        specialties = []
        for line in lines:
            # 번호 제거 및 정제
            specialty = line.strip().lstrip('0123456789.-) ')
            if specialty:
                specialties.append(specialty)
        return specialties[:5]


class DynamicAgent:
    """동적으로 생성된 에이전트"""

    def __init__(
        self,
        agent_id: str,
        specialty: str,
        system_prompt: str,
        tools: list
    ):
        self.agent_id = agent_id
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    def invoke(self, task: str) -> str:
        """에이전트 실행"""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{task}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"task": task})

        return response.content
```

---

## 8단계: 모니터링 및 디버깅

### 8.1 monitoring.py

```python
# app/monitoring.py
from langsmith import Client
from datetime import datetime, timedelta
from typing import Dict, List


class MultiAgentMonitor:
    """멀티 에이전트 시스템 모니터링"""

    def __init__(self, project_name: str = "multi-agent-tutorial"):
        self.client = Client()
        self.project_name = project_name

    def get_agent_performance(self, hours: int = 24) -> Dict:
        """에이전트별 성능 분석"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours)
        ))

        agent_stats = {
            "researcher": {"count": 0, "avg_latency": 0, "errors": 0},
            "writer": {"count": 0, "avg_latency": 0, "errors": 0},
            "reviewer": {"count": 0, "avg_latency": 0, "errors": 0},
            "supervisor": {"count": 0, "avg_latency": 0, "errors": 0}
        }

        for run in runs:
            name = run.name.lower()
            for agent in agent_stats:
                if agent in name:
                    agent_stats[agent]["count"] += 1
                    if run.end_time and run.start_time:
                        latency = (run.end_time - run.start_time).total_seconds()
                        current_avg = agent_stats[agent]["avg_latency"]
                        count = agent_stats[agent]["count"]
                        agent_stats[agent]["avg_latency"] = (
                            (current_avg * (count - 1) + latency) / count
                        )
                    if run.error:
                        agent_stats[agent]["errors"] += 1
                    break

        return agent_stats

    def get_workflow_metrics(self, hours: int = 24) -> Dict:
        """워크플로우 메트릭"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),
            is_root=True
        ))

        if not runs:
            return {"message": "데이터 없음"}

        total = len(runs)
        successful = sum(1 for r in runs if not r.error)
        total_latency = sum(
            (r.end_time - r.start_time).total_seconds()
            for r in runs
            if r.end_time and r.start_time
        )

        return {
            "total_workflows": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_workflow_time": total_latency / total if total > 0 else 0
        }

    def print_dashboard(self, hours: int = 24):
        """대시보드 출력"""
        agent_perf = self.get_agent_performance(hours)
        workflow_metrics = self.get_workflow_metrics(hours)

        print("\n" + "="*60)
        print(f"📊 멀티 에이전트 시스템 대시보드 (최근 {hours}시간)")
        print("="*60)

        print("\n📈 워크플로우 메트릭:")
        print(f"  총 실행 수: {workflow_metrics.get('total_workflows', 0)}")
        print(f"  성공률: {workflow_metrics.get('success_rate', 0):.1%}")
        print(f"  평균 실행 시간: {workflow_metrics.get('avg_workflow_time', 0):.1f}초")

        print("\n👥 에이전트별 성능:")
        for agent, stats in agent_perf.items():
            print(f"\n  {agent.capitalize()}:")
            print(f"    - 호출 수: {stats['count']}")
            print(f"    - 평균 지연: {stats['avg_latency']:.2f}초")
            print(f"    - 에러 수: {stats['errors']}")

        print("\n" + "="*60)


# 사용 예시
if __name__ == "__main__":
    monitor = MultiAgentMonitor()
    monitor.print_dashboard(hours=24)
```

---

## 9단계: 테스트

### 9.1 tests/test_workflow.py

```python
# tests/test_workflow.py
import pytest
from graph.workflow import MultiAgentWorkflow
from graph.state import create_initial_state


class TestMultiAgentWorkflow:
    """멀티 에이전트 워크플로우 테스트"""

    @pytest.fixture
    def workflow(self):
        w = MultiAgentWorkflow()
        w.compile(with_memory=False)
        return w

    def test_workflow_initialization(self, workflow):
        """워크플로우 초기화 테스트"""
        assert workflow.supervisor is not None
        assert workflow.researcher is not None
        assert workflow.writer is not None
        assert workflow.reviewer is not None
        assert workflow.app is not None

    def test_initial_state_creation(self):
        """초기 상태 생성 테스트"""
        state = create_initial_state("테스트 작업")

        assert state["task"] == "테스트 작업"
        assert state["next_agent"] == "supervisor"
        assert state["iteration_count"] == 0

    def test_simple_workflow_run(self, workflow):
        """간단한 워크플로우 실행 테스트"""
        result = workflow.run(
            "AI의 간단한 설명을 작성해주세요.",
            thread_id="test-1"
        )

        assert "status" in result
        assert "iterations" in result
        assert result["iterations"] > 0

    def test_workflow_streaming(self, workflow):
        """스트리밍 실행 테스트"""
        events = list(workflow.stream(
            "테스트 콘텐츠 작성",
            thread_id="test-2"
        ))

        assert len(events) > 0


class TestAgents:
    """개별 에이전트 테스트"""

    def test_researcher_agent(self):
        """연구 에이전트 테스트"""
        from agents.researcher import ResearcherAgent

        researcher = ResearcherAgent()
        assert researcher.name == "Researcher"

        state = {
            "task": "Python 프로그래밍에 대해 조사",
            "messages": []
        }

        result = researcher.invoke(state)
        assert "research_data" in result
        assert "messages" in result

    def test_writer_agent(self):
        """작성 에이전트 테스트"""
        from agents.writer import WriterAgent

        writer = WriterAgent()
        assert writer.name == "Writer"

        state = {
            "task": "Python 소개 글 작성",
            "messages": [],
            "research_data": {
                "topic": "Python",
                "findings": "Python은 인기 있는 프로그래밍 언어입니다.",
                "sources": [],
                "confidence": 0.9
            }
        }

        result = writer.invoke(state)
        assert "draft_content" in result

    def test_reviewer_agent(self):
        """검토 에이전트 테스트"""
        from agents.reviewer import ReviewerAgent

        reviewer = ReviewerAgent()
        assert reviewer.name == "Reviewer"

        state = {
            "task": "글 검토",
            "messages": [],
            "draft_content": {
                "title": "Python 소개",
                "sections": [{"title": "소개", "content": "..."}],
                "word_count": 500
            }
        }

        result = reviewer.invoke(state)
        assert "review_feedback" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 마무리

### 학습한 내용

1. ✅ 멀티 에이전트 시스템 설계 원칙
2. ✅ LangGraph를 활용한 에이전트 간 협업
3. ✅ 도구 사용 에이전트 구현
4. ✅ 슈퍼바이저 패턴 적용
5. ✅ 계층적 에이전트 구조
6. ✅ LangSmith를 통한 모니터링

### 다음 단계

- [튜토리얼 3: 대화형 AI 어시스턴트](./03-conversational-assistant.md)

### 추가 개선 아이디어

1. **에이전트 자기 반성**: 에이전트가 자신의 출력을 평가하고 개선
2. **동적 팀 구성**: 작업에 따라 필요한 에이전트 자동 선정
3. **에이전트 간 직접 통신**: 슈퍼바이저 없이 에이전트끼리 직접 협업
4. **장기 메모리**: 세션 간 학습 내용 유지
5. **병렬 에이전트 실행**: 독립적인 작업 병렬 처리
