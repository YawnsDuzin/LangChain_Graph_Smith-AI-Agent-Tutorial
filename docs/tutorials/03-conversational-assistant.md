# 튜토리얼 3: 대화형 AI 어시스턴트 구축

## 개요

이 튜토리얼에서는 **도구 사용 능력을 갖춘 고급 대화형 AI 어시스턴트**를 구축합니다. 실시간 정보 검색, 코드 실행, 파일 처리 등 다양한 기능을 갖춘 실용적인 어시스턴트를 만들어봅니다.

### 학습 목표

- 도구 사용 에이전트 설계 및 구현
- Human-in-the-Loop 패턴 적용
- 스트리밍 응답 구현
- 대화 메모리 관리
- 프로덕션 배포 고려사항

### 최종 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                 대화형 AI 어시스턴트 아키텍처                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   사용자 입력                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────┐                                           │
│   │   입력 처리      │                                           │
│   │  (Preprocessing)│                                           │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐      ┌──────────────────────────────┐    │
│   │   의도 분석      │      │         도구 목록            │    │
│   │  (Intent)       │      │  🔍 웹 검색                  │    │
│   └────────┬────────┘      │  🧮 계산기                   │    │
│            │               │  🐍 Python 실행              │    │
│    ┌───────┴───────┐       │  📁 파일 처리                │    │
│    │               │       │  🌤️ 날씨 조회               │    │
│    ▼               ▼       │  📅 일정 관리                │    │
│ ┌──────┐      ┌──────┐     └──────────────────────────────┘    │
│ │ 대화  │      │ 도구  │                                        │
│ │ 응답  │      │ 사용  │◄───────────────────────────────────┐   │
│ └──┬───┘      └──┬───┘                                     │   │
│    │             │                                          │   │
│    │             ▼                                          │   │
│    │      ┌─────────────┐    승인 필요?                     │   │
│    │      │ Human-in-   │◄──────────────┐                  │   │
│    │      │ the-Loop    │               │                  │   │
│    │      └──────┬──────┘               │                  │   │
│    │             │                      │                  │   │
│    │             ▼                      │                  │   │
│    │      ┌─────────────┐               │                  │   │
│    │      │  도구 실행   │───────────────┘                  │   │
│    │      └──────┬──────┘                                  │   │
│    │             │                                          │   │
│    └──────┬──────┘                                          │   │
│           │                                                 │   │
│           ▼                                                 │   │
│    ┌─────────────┐                                          │   │
│    │  응답 생성   │──────────────────────────────────────────┘   │
│    └──────┬──────┘  (도구 결과 반영)                            │
│           │                                                     │
│           ▼                                                     │
│    ┌─────────────┐     ┌─────────────┐                         │
│    │  스트리밍    │────▶│ LangSmith   │                         │
│    │  출력       │     │  추적       │                         │
│    └─────────────┘     └─────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1단계: 프로젝트 설정

### 1.1 디렉토리 구조

```
ai-assistant/
├── assistant/
│   ├── __init__.py
│   ├── config.py           # 설정 관리
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py       # 검색 도구
│   │   ├── calculator.py   # 계산 도구
│   │   ├── code_executor.py # 코드 실행
│   │   ├── file_handler.py # 파일 처리
│   │   └── utilities.py    # 유틸리티 도구
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation.py # 대화 메모리
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py        # 상태 정의
│   │   └── assistant.py    # 어시스턴트 그래프
│   └── main.py             # 메인 앱
├── web/
│   ├── app.py              # FastAPI 앱
│   └── templates/
│       └── chat.html       # 웹 UI
├── tests/
│   └── test_assistant.py
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
fastapi>=0.109.0
uvicorn>=0.27.0
websockets>=12.0
python-dateutil>=2.8.0
numexpr>=2.8.0
```

### 1.3 환경 변수

```bash
# .env
OPENAI_API_KEY=sk-your-openai-api-key
TAVILY_API_KEY=tvly-your-tavily-key
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-assistant-tutorial

# 선택적
WEATHER_API_KEY=your-weather-api-key
```

---

## 2단계: 도구 구현

### 2.1 tools/search.py

```python
# assistant/tools/search.py
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional
import os


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    인터넷에서 최신 정보를 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본값: 5)

    Returns:
        검색 결과 요약
    """
    try:
        search = TavilySearchResults(max_results=max_results)
        results = search.invoke(query)

        if not results:
            return "검색 결과가 없습니다."

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No Title')
            url = result.get('url', 'N/A')
            content = result.get('content', '')[:300]
            formatted.append(f"[{i}] {title}\n    URL: {url}\n    {content}...")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"


@tool
def news_search(topic: str, days: int = 7) -> str:
    """
    최근 뉴스를 검색합니다.

    Args:
        topic: 뉴스 주제
        days: 검색할 기간 (일 단위)

    Returns:
        관련 뉴스 목록
    """
    query = f"{topic} 뉴스 최근 {days}일"
    return web_search.invoke({"query": query, "max_results": 5})


@tool
def wikipedia_search(topic: str) -> str:
    """
    위키피디아에서 정보를 검색합니다.

    Args:
        topic: 검색할 주제

    Returns:
        위키피디아 정보 요약
    """
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        result = wikipedia.run(topic)
        return result[:2000]  # 길이 제한

    except Exception as e:
        return f"위키피디아 검색 오류: {str(e)}"
```

### 2.2 tools/calculator.py

```python
# assistant/tools/calculator.py
from langchain_core.tools import tool
import numexpr as ne
import math
from typing import Union


@tool
def calculate(expression: str) -> str:
    """
    수학 표현식을 계산합니다.

    Args:
        expression: 계산할 수학 표현식 (예: "2 + 2", "sqrt(16)", "sin(3.14)")

    Returns:
        계산 결과
    """
    try:
        # 안전한 수학 함수 매핑
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
        }

        # numexpr로 계산 (더 복잡한 표현식 지원)
        result = ne.evaluate(expression).item()
        return f"계산 결과: {expression} = {result}"

    except Exception as e:
        try:
            # 폴백: eval with restricted globals
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"계산 결과: {expression} = {result}"
        except Exception as e2:
            return f"계산 오류: {str(e2)}"


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """
    단위를 변환합니다.

    Args:
        value: 변환할 값
        from_unit: 원래 단위 (예: km, mile, kg, lb, celsius, fahrenheit)
        to_unit: 변환할 단위

    Returns:
        변환된 값
    """
    conversions = {
        # 길이
        ("km", "mile"): lambda x: x * 0.621371,
        ("mile", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("cm", "inch"): lambda x: x * 0.393701,
        ("inch", "cm"): lambda x: x * 2.54,

        # 무게
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("g", "oz"): lambda x: x * 0.035274,
        ("oz", "g"): lambda x: x * 28.3495,

        # 온도
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("celsius", "kelvin"): lambda x: x + 273.15,
        ("kelvin", "celsius"): lambda x: x - 273.15,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"지원하지 않는 단위 변환입니다: {from_unit} → {to_unit}"


@tool
def percentage_calculate(operation: str, value1: float, value2: float = None) -> str:
    """
    퍼센트 관련 계산을 수행합니다.

    Args:
        operation: 계산 유형 ('of', 'change', 'increase', 'decrease')
        value1: 첫 번째 값
        value2: 두 번째 값 (필요한 경우)

    Returns:
        계산 결과
    """
    try:
        if operation == "of" and value2:
            # value1%의 value2
            result = (value1 / 100) * value2
            return f"{value1}%의 {value2} = {result}"

        elif operation == "change" and value2:
            # value1에서 value2로의 변화율
            change = ((value2 - value1) / value1) * 100
            return f"{value1}에서 {value2}로의 변화율 = {change:.2f}%"

        elif operation == "increase" and value2:
            # value1을 value2% 증가
            result = value1 * (1 + value2/100)
            return f"{value1}을 {value2}% 증가 = {result}"

        elif operation == "decrease" and value2:
            # value1을 value2% 감소
            result = value1 * (1 - value2/100)
            return f"{value1}을 {value2}% 감소 = {result}"

        else:
            return "지원하지 않는 연산입니다. 'of', 'change', 'increase', 'decrease' 중 선택하세요."

    except Exception as e:
        return f"계산 오류: {str(e)}"
```

### 2.3 tools/code_executor.py

```python
# assistant/tools/code_executor.py
from langchain_core.tools import tool
import subprocess
import tempfile
import os
from typing import Optional


@tool
def execute_python(code: str, timeout: int = 30) -> str:
    """
    Python 코드를 실행합니다. 주의: 이 도구는 안전한 환경에서만 사용하세요.

    Args:
        code: 실행할 Python 코드
        timeout: 실행 제한 시간 (초)

    Returns:
        실행 결과 또는 에러 메시지
    """
    # 위험한 명령어 필터링
    dangerous_patterns = [
        "import os", "import subprocess", "import sys",
        "__import__", "eval(", "exec(", "open(",
        "shutil", "rmdir", "remove", "unlink"
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return f"보안 오류: '{pattern}' 사용이 제한됩니다."

    try:
        # 임시 파일에 코드 저장
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        # 코드 실행
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # 임시 파일 삭제
        os.unlink(temp_file)

        output = result.stdout
        if result.stderr:
            output += f"\n에러:\n{result.stderr}"

        return output if output.strip() else "코드가 성공적으로 실행되었습니다. (출력 없음)"

    except subprocess.TimeoutExpired:
        return f"실행 시간 초과 ({timeout}초)"
    except Exception as e:
        return f"실행 오류: {str(e)}"


@tool
def explain_code(code: str, language: str = "python") -> str:
    """
    코드를 분석하고 설명합니다.

    Args:
        code: 분석할 코드
        language: 프로그래밍 언어

    Returns:
        코드 설명
    """
    # 이 함수는 실제로 LLM을 호출하여 설명을 생성해야 합니다
    # 여기서는 간단한 구조 분석만 수행
    lines = code.strip().split('\n')
    analysis = {
        "총 라인 수": len(lines),
        "함수 정의": sum(1 for l in lines if l.strip().startswith('def ')),
        "클래스 정의": sum(1 for l in lines if l.strip().startswith('class ')),
        "주석": sum(1 for l in lines if l.strip().startswith('#')),
        "import 문": sum(1 for l in lines if l.strip().startswith('import ') or l.strip().startswith('from '))
    }

    result = f"코드 분석 결과 ({language}):\n"
    for key, value in analysis.items():
        result += f"  - {key}: {value}\n"

    return result


@tool
def generate_code(description: str, language: str = "python") -> str:
    """
    설명을 바탕으로 코드 템플릿을 생성합니다.

    Args:
        description: 생성할 코드에 대한 설명
        language: 프로그래밍 언어

    Returns:
        생성된 코드 템플릿
    """
    # 이 함수는 LLM을 통해 코드를 생성해야 합니다
    # 여기서는 간단한 템플릿 반환
    templates = {
        "함수": '''
def example_function(param1, param2):
    """
    {description}

    Args:
        param1: 첫 번째 매개변수
        param2: 두 번째 매개변수

    Returns:
        결과 값
    """
    # TODO: 구현
    pass
''',
        "클래스": '''
class ExampleClass:
    """
    {description}
    """

    def __init__(self):
        """초기화"""
        pass

    def method(self):
        """메서드"""
        pass
'''
    }

    if "class" in description.lower() or "클래스" in description:
        return templates["클래스"].format(description=description)
    else:
        return templates["함수"].format(description=description)
```

### 2.4 tools/utilities.py

```python
# assistant/tools/utilities.py
from langchain_core.tools import tool
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import json


@tool
def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """
    현재 시간을 반환합니다.

    Args:
        timezone: 시간대 (기본값: Asia/Seoul)

    Returns:
        현재 날짜와 시간
    """
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S (%Z)")
    except:
        now = datetime.now()
        return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S")


@tool
def calculate_date(
    operation: str,
    days: int = 0,
    weeks: int = 0,
    months: int = 0,
    base_date: str = None
) -> str:
    """
    날짜를 계산합니다.

    Args:
        operation: 'add' 또는 'subtract'
        days: 일 수
        weeks: 주 수
        months: 월 수
        base_date: 기준 날짜 (없으면 오늘)

    Returns:
        계산된 날짜
    """
    try:
        if base_date:
            base = date_parser.parse(base_date)
        else:
            base = datetime.now()

        total_days = days + (weeks * 7) + (months * 30)

        if operation == "add":
            result = base + timedelta(days=total_days)
        elif operation == "subtract":
            result = base - timedelta(days=total_days)
        else:
            return "operation은 'add' 또는 'subtract'여야 합니다."

        return f"결과: {result.strftime('%Y년 %m월 %d일 %A')}"

    except Exception as e:
        return f"날짜 계산 오류: {str(e)}"


@tool
def days_between(date1: str, date2: str) -> str:
    """
    두 날짜 사이의 일수를 계산합니다.

    Args:
        date1: 첫 번째 날짜
        date2: 두 번째 날짜

    Returns:
        날짜 차이
    """
    try:
        d1 = date_parser.parse(date1)
        d2 = date_parser.parse(date2)
        diff = abs((d2 - d1).days)
        return f"{date1}와 {date2} 사이: {diff}일"
    except Exception as e:
        return f"날짜 파싱 오류: {str(e)}"


@tool
def format_json(json_string: str, indent: int = 2) -> str:
    """
    JSON 문자열을 보기 좋게 포맷팅합니다.

    Args:
        json_string: JSON 문자열
        indent: 들여쓰기 크기

    Returns:
        포맷팅된 JSON
    """
    try:
        data = json.loads(json_string)
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except Exception as e:
        return f"JSON 파싱 오류: {str(e)}"


@tool
def translate_text(text: str, target_language: str = "en") -> str:
    """
    텍스트를 번역합니다. (시뮬레이션)

    Args:
        text: 번역할 텍스트
        target_language: 대상 언어 코드 (en, ko, ja, zh 등)

    Returns:
        번역된 텍스트
    """
    # 실제로는 번역 API를 호출해야 합니다
    return f"[{target_language}로 번역 요청됨]\n원본: {text}\n(실제 번역은 번역 API 연동이 필요합니다)"


@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    텍스트를 요약합니다.

    Args:
        text: 요약할 텍스트
        max_length: 최대 요약 길이

    Returns:
        요약된 텍스트
    """
    # 간단한 추출 요약 (실제로는 LLM 활용)
    sentences = text.split('.')
    if len(sentences) <= 3:
        return text

    # 첫 3문장 추출
    summary = '. '.join(sentences[:3]) + '.'
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return f"요약:\n{summary}"


# 모든 유틸리티 도구
utility_tools = [
    get_current_time,
    calculate_date,
    days_between,
    format_json,
    translate_text,
    summarize_text
]
```

### 2.5 tools/__init__.py

```python
# assistant/tools/__init__.py
from .search import web_search, news_search, wikipedia_search
from .calculator import calculate, unit_convert, percentage_calculate
from .code_executor import execute_python, explain_code, generate_code
from .utilities import (
    get_current_time,
    calculate_date,
    days_between,
    format_json,
    translate_text,
    summarize_text
)

# 모든 도구 목록
ALL_TOOLS = [
    # 검색 도구
    web_search,
    news_search,
    wikipedia_search,

    # 계산 도구
    calculate,
    unit_convert,
    percentage_calculate,

    # 코드 도구
    execute_python,
    explain_code,
    generate_code,

    # 유틸리티 도구
    get_current_time,
    calculate_date,
    days_between,
    format_json,
    translate_text,
    summarize_text,
]

# 카테고리별 도구
TOOL_CATEGORIES = {
    "검색": [web_search, news_search, wikipedia_search],
    "계산": [calculate, unit_convert, percentage_calculate],
    "코드": [execute_python, explain_code, generate_code],
    "유틸리티": [get_current_time, calculate_date, days_between, format_json],
}

# 승인이 필요한 도구 (Human-in-the-Loop)
TOOLS_REQUIRING_APPROVAL = [
    "execute_python",  # 코드 실행은 승인 필요
]
```

---

## 3단계: 대화 메모리 구현

### 3.1 memory/conversation.py

```python
# assistant/memory/conversation.py
from typing import List, Dict, Optional, Any
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.chat_history import BaseChatMessageHistory
from datetime import datetime
import json


class ConversationMemory(BaseChatMessageHistory):
    """대화 메모리 관리 클래스"""

    def __init__(
        self,
        session_id: str,
        max_messages: int = 50,
        max_tokens: int = 4000
    ):
        self.session_id = session_id
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: List[BaseMessage] = []
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "message_count": 0
        }

    @property
    def messages(self) -> List[BaseMessage]:
        """메시지 목록 반환"""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """메시지 추가"""
        self._messages.append(message)
        self._metadata["message_count"] += 1

        # 최대 메시지 수 초과 시 오래된 메시지 제거
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def add_user_message(self, message: str) -> None:
        """사용자 메시지 추가"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """AI 메시지 추가"""
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """대화 기록 초기화"""
        self._messages = []
        self._metadata["message_count"] = 0

    def get_recent_messages(self, n: int = 10) -> List[BaseMessage]:
        """최근 n개 메시지 반환"""
        return self._messages[-n:]

    def get_summary(self) -> str:
        """대화 요약 반환"""
        if not self._messages:
            return "대화 기록이 없습니다."

        summary = f"세션 ID: {self.session_id}\n"
        summary += f"총 메시지 수: {len(self._messages)}\n"
        summary += f"생성 시간: {self._metadata['created_at']}\n"

        return summary

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "messages": [
                {
                    "type": type(m).__name__,
                    "content": m.content
                }
                for m in self._messages
            ],
            "metadata": self._metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMemory":
        """딕셔너리에서 복원"""
        memory = cls(session_id=data["session_id"])
        memory._metadata = data.get("metadata", {})

        for msg_data in data.get("messages", []):
            msg_type = msg_data["type"]
            content = msg_data["content"]

            if msg_type == "HumanMessage":
                memory.add_message(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                memory.add_message(AIMessage(content=content))
            elif msg_type == "SystemMessage":
                memory.add_message(SystemMessage(content=content))

        return memory

    def save_to_file(self, filepath: str) -> None:
        """파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ConversationMemory":
        """파일에서 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ConversationStore:
    """여러 대화 세션 관리"""

    def __init__(self):
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_session(self, session_id: str) -> ConversationMemory:
        """세션 가져오기 (없으면 생성)"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """모든 세션 ID 목록"""
        return list(self._sessions.keys())

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """세션 요약"""
        if session_id in self._sessions:
            return self._sessions[session_id].get_summary()
        return None


# 전역 대화 저장소
conversation_store = ConversationStore()
```

---

## 4단계: 어시스턴트 그래프 구현

### 4.1 graph/state.py

```python
# assistant/graph/state.py
from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ToolCall(TypedDict):
    """도구 호출 정보"""
    tool_name: str
    tool_args: dict
    requires_approval: bool
    approved: Optional[bool]


class AssistantState(TypedDict):
    """어시스턴트 상태"""
    # 대화 메시지
    messages: Annotated[list, add_messages]

    # 현재 사용자 입력
    user_input: str

    # 의도 분석 결과
    intent: str  # chat, tool_use, clarify

    # 도구 관련
    pending_tool_calls: List[ToolCall]
    tool_results: List[dict]

    # 워크플로우 제어
    requires_approval: bool
    user_approved: Optional[bool]

    # 응답
    response: str
    streaming: bool

    # 세션 정보
    session_id: str


def create_initial_state(
    user_input: str,
    session_id: str = "default"
) -> AssistantState:
    """초기 상태 생성"""
    return AssistantState(
        messages=[],
        user_input=user_input,
        intent="",
        pending_tool_calls=[],
        tool_results=[],
        requires_approval=False,
        user_approved=None,
        response="",
        streaming=True,
        session_id=session_id
    )
```

### 4.2 graph/assistant.py

```python
# assistant/graph/assistant.py
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable

from assistant.graph.state import AssistantState, create_initial_state
from assistant.tools import ALL_TOOLS, TOOLS_REQUIRING_APPROVAL
from assistant.memory.conversation import conversation_store


class AIAssistant:
    """대화형 AI 어시스턴트"""

    SYSTEM_PROMPT = """당신은 유능하고 친절한 AI 어시스턴트입니다.

당신의 능력:
1. 🔍 인터넷 검색으로 최신 정보 제공
2. 🧮 수학 계산 및 단위 변환
3. 🐍 Python 코드 실행 및 설명
4. 📅 날짜/시간 계산
5. 📝 텍스트 요약 및 번역

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
        self.model = model
        self.temperature = temperature
        self.tools = ALL_TOOLS
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True
        ).bind_tools(self.tools)

        self.graph = self._build_graph()
        self.app = None

    def _build_graph(self) -> StateGraph:
        """그래프 구축"""
        graph = StateGraph(AssistantState)

        # 노드 추가
        graph.add_node("process_input", self._process_input)
        graph.add_node("assistant", self._assistant_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("check_approval", self._check_approval)
        graph.add_node("generate_response", self._generate_response)

        # 엣지 추가
        graph.add_edge(START, "process_input")
        graph.add_edge("process_input", "assistant")

        # 어시스턴트 후 조건부 라우팅
        graph.add_conditional_edges(
            "assistant",
            self._route_after_assistant,
            {
                "tools": "tools",
                "check_approval": "check_approval",
                "respond": "generate_response"
            }
        )

        # 도구 실행 후 어시스턴트로 복귀
        graph.add_edge("tools", "assistant")

        # 승인 확인 후 라우팅
        graph.add_conditional_edges(
            "check_approval",
            self._route_after_approval,
            {
                "tools": "tools",
                "respond": "generate_response"
            }
        )

        graph.add_edge("generate_response", END)

        return graph

    @traceable(name="process_input")
    def _process_input(self, state: AssistantState) -> dict:
        """입력 처리"""
        user_input = state["user_input"]
        session_id = state["session_id"]

        # 대화 기록 가져오기
        memory = conversation_store.get_session(session_id)
        memory.add_user_message(user_input)

        return {
            "messages": [HumanMessage(content=user_input)]
        }

    @traceable(name="assistant")
    def _assistant_node(self, state: AssistantState) -> dict:
        """어시스턴트 노드"""
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]

        response = self.llm.invoke(messages)

        return {"messages": [response]}

    def _route_after_assistant(
        self,
        state: AssistantState
    ) -> Literal["tools", "check_approval", "respond"]:
        """어시스턴트 후 라우팅"""
        last_message = state["messages"][-1]

        # 도구 호출이 있는지 확인
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
        """승인 확인 노드"""
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
        """승인 후 라우팅"""
        if state.get("user_approved"):
            return "tools"
        return "respond"

    @traceable(name="generate_response")
    def _generate_response(self, state: AssistantState) -> dict:
        """최종 응답 생성"""
        last_message = state["messages"][-1]

        if hasattr(last_message, "content"):
            response = last_message.content
        else:
            response = str(last_message)

        # 대화 기록에 추가
        session_id = state["session_id"]
        memory = conversation_store.get_session(session_id)
        memory.add_ai_message(response)

        return {"response": response}

    def compile(self, with_memory: bool = True):
        """그래프 컴파일"""
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
        """동기식 채팅"""
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
        """비동기식 채팅"""
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
        """스트리밍 채팅"""
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(message, session_id)
        config = {"configurable": {"thread_id": session_id}}

        for event in self.app.stream(
            initial_state,
            config=config,
            stream_mode="values"
        ):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    yield {
                        "type": "message",
                        "content": last_msg.content
                    }

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
        """도구 실행 승인"""
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
        """대화 기록 가져오기"""
        memory = conversation_store.get_session(session_id)
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in memory.messages
        ]

    def clear_conversation(self, session_id: str):
        """대화 기록 초기화"""
        memory = conversation_store.get_session(session_id)
        memory.clear()


# 사용 예시
if __name__ == "__main__":
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
```

---

## 5단계: 웹 인터페이스 구현

### 5.1 web/app.py

```python
# web/app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import asyncio

from assistant.graph.assistant import AIAssistant

app = FastAPI(title="AI Assistant API")
assistant = AIAssistant()
assistant.compile()

# 연결된 WebSocket 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = ConnectionManager()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


# REST API 엔드포인트
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 API"""
    try:
        response = await assistant.achat(
            request.message,
            session_id=request.session_id
        )
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/approve")
async def approve_endpoint(request: ApprovalRequest):
    """도구 실행 승인 API"""
    assistant.approve_tool_execution(
        request.session_id,
        request.approved
    )
    return {"status": "approved" if request.approved else "rejected"}


@app.get("/api/history/{session_id}")
async def history_endpoint(session_id: str):
    """대화 기록 API"""
    history = assistant.get_conversation_history(session_id)
    return {"history": history}


@app.delete("/api/history/{session_id}")
async def clear_history_endpoint(session_id: str):
    """대화 기록 삭제 API"""
    assistant.clear_conversation(session_id)
    return {"status": "cleared"}


# WebSocket 엔드포인트
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 채팅"""
    await manager.connect(websocket, session_id)

    try:
        while True:
            # 메시지 수신
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data.get("type") == "chat":
                user_message = message_data.get("message", "")

                # 스트리밍 응답
                async for event in stream_response(user_message, session_id):
                    await manager.send_message(event, session_id)

            elif message_data.get("type") == "approve":
                approved = message_data.get("approved", False)
                assistant.approve_tool_execution(session_id, approved)
                await manager.send_message(
                    {"type": "approval_processed", "approved": approved},
                    session_id
                )

    except WebSocketDisconnect:
        manager.disconnect(session_id)


async def stream_response(message: str, session_id: str):
    """스트리밍 응답 생성"""
    for event in assistant.stream_chat(message, session_id):
        yield event
        await asyncio.sleep(0.01)  # 약간의 지연

    yield {"type": "done"}


# HTML 페이지
@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .chat-container {
            width: 100%;
            max-width: 800px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .chat-header p {
            opacity: 0.8;
            font-size: 14px;
        }

        .chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }

        .message.user {
            align-items: flex-end;
        }

        .message.assistant {
            align-items: flex-start;
        }

        .message-content {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .tool-call {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 14px;
        }

        .chat-input {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
        }

        .chat-input input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #eee;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input input:focus {
            border-color: #667eea;
        }

        .chat-input button {
            margin-left: 10px;
            padding: 15px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .chat-input button:hover {
            transform: scale(1.05);
        }

        .typing-indicator {
            display: none;
            padding: 10px;
        }

        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 AI Assistant</h1>
            <p>무엇이든 물어보세요!</p>
        </div>

        <div class="chat-messages" id="messages">
            <div class="message assistant">
                <div class="message-content">
                    안녕하세요! 저는 AI 어시스턴트입니다. 웹 검색, 계산, 코드 실행 등 다양한 작업을 도와드릴 수 있습니다. 무엇을 도와드릴까요?
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typing">
            <span></span>
            <span></span>
            <span></span>
        </div>

        <div class="chat-input">
            <input type="text" id="input" placeholder="메시지를 입력하세요..." autocomplete="off">
            <button onclick="sendMessage()">전송</button>
        </div>
    </div>

    <script>
        const sessionId = 'session-' + Math.random().toString(36).substr(2, 9);
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('input');
        const typing = document.getElementById('typing');

        // WebSocket 연결
        const ws = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.type === 'message') {
                typing.style.display = 'none';
                addMessage(data.content, 'assistant');
            } else if (data.type === 'tool_call') {
                addToolCall(data.calls);
            } else if (data.type === 'done') {
                typing.style.display = 'none';
            }
        };

        function sendMessage() {
            const message = input.value.trim();
            if (!message) return;

            addMessage(message, 'user');
            input.value = '';
            typing.style.display = 'block';

            ws.send(JSON.stringify({
                type: 'chat',
                message: message
            }));
        }

        function addMessage(content, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.innerHTML = `<div class="message-content">${formatContent(content)}</div>`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function addToolCall(calls) {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'tool-call';
            toolDiv.innerHTML = `🔧 도구 호출: ${calls.map(c => c.name).join(', ')}`;
            messagesDiv.appendChild(toolDiv);
        }

        function formatContent(content) {
            // 간단한 마크다운 변환
            return content
                .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }

        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 6단계: 메인 애플리케이션

### 6.1 main.py

```python
# assistant/main.py
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from assistant.graph.assistant import AIAssistant
from assistant.memory.conversation import conversation_store


def print_header():
    print("\n" + "="*60)
    print("🤖 대화형 AI 어시스턴트")
    print("="*60)
    print("\n사용 가능한 기능:")
    print("  🔍 웹 검색: '~에 대해 검색해줘'")
    print("  🧮 계산기: '123 * 456 계산해줘'")
    print("  🐍 코드 실행: 'Python으로 ~해줘'")
    print("  📅 날짜/시간: '오늘 날짜가 뭐야?'")
    print("  📝 유틸리티: '텍스트 요약해줘'")
    print("\n명령어:")
    print("  /clear - 대화 기록 삭제")
    print("  /history - 대화 기록 보기")
    print("  /help - 도움말")
    print("  quit 또는 exit - 종료")
    print("-"*60 + "\n")


def print_help():
    print("""
📚 AI 어시스턴트 도움말

[검색 기능]
• "최신 AI 뉴스 알려줘" - 웹 검색
• "파이썬이 뭐야?" - 위키피디아 검색

[계산 기능]
• "123 + 456 계산해줘" - 기본 계산
• "100달러를 원으로" - 단위 변환
• "20%의 500은?" - 퍼센트 계산

[코드 기능]
• "피보나치 함수 만들어줘" - 코드 생성
• "이 코드 설명해줘" - 코드 설명
• "이 코드 실행해줘" - 코드 실행 (승인 필요)

[날짜/시간]
• "지금 몇 시야?" - 현재 시간
• "오늘부터 100일 후는?" - 날짜 계산

[유틸리티]
• "이 텍스트 요약해줘" - 텍스트 요약
• "영어로 번역해줘" - 번역 (시뮬레이션)
""")


def run_cli():
    """CLI 모드 실행"""
    print_header()

    assistant = AIAssistant()
    assistant.compile()

    session_id = "cli-session"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # 명령어 처리
            if user_input.startswith("/"):
                cmd = user_input[1:].lower()

                if cmd == "clear":
                    assistant.clear_conversation(session_id)
                    print("✅ 대화 기록이 삭제되었습니다.\n")
                    continue

                elif cmd == "history":
                    history = assistant.get_conversation_history(session_id)
                    if not history:
                        print("대화 기록이 없습니다.\n")
                    else:
                        print("\n📜 대화 기록:")
                        for msg in history:
                            role = "You" if msg["role"] == "user" else "AI"
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"  {role}: {content}")
                        print()
                    continue

                elif cmd == "help":
                    print_help()
                    continue

                else:
                    print(f"알 수 없는 명령어: /{cmd}\n")
                    continue

            # 종료
            if user_input.lower() in ["quit", "exit", "종료"]:
                print("\n👋 안녕히 가세요!")
                break

            # 응답 생성
            print("\n", end="")
            response = assistant.chat(user_input, session_id=session_id)
            print(f"AI: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")


def run_web():
    """웹 서버 실행"""
    print("\n🌐 웹 서버를 시작합니다...")
    print("브라우저에서 http://localhost:8000 을 열어주세요.\n")

    import uvicorn
    from web.app import app
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "web":
            run_web()
        elif mode == "cli":
            run_cli()
        else:
            print(f"알 수 없는 모드: {mode}")
            print("사용법: python main.py [cli|web]")
    else:
        # 기본: CLI 모드
        run_cli()


if __name__ == "__main__":
    main()
```

---

## 7단계: 테스트

### 7.1 tests/test_assistant.py

```python
# tests/test_assistant.py
import pytest
from assistant.graph.assistant import AIAssistant
from assistant.tools import calculate, web_search, get_current_time
from assistant.memory.conversation import ConversationMemory, conversation_store


class TestTools:
    """도구 테스트"""

    def test_calculate(self):
        """계산 도구 테스트"""
        result = calculate.invoke({"expression": "2 + 2"})
        assert "4" in result

    def test_get_current_time(self):
        """시간 도구 테스트"""
        result = get_current_time.invoke({})
        assert "년" in result
        assert "월" in result


class TestConversationMemory:
    """대화 메모리 테스트"""

    def test_add_messages(self):
        """메시지 추가 테스트"""
        memory = ConversationMemory("test-session")
        memory.add_user_message("안녕하세요")
        memory.add_ai_message("안녕하세요! 무엇을 도와드릴까요?")

        assert len(memory.messages) == 2

    def test_clear_messages(self):
        """메시지 초기화 테스트"""
        memory = ConversationMemory("test-session")
        memory.add_user_message("테스트")
        memory.clear()

        assert len(memory.messages) == 0

    def test_conversation_store(self):
        """대화 저장소 테스트"""
        session = conversation_store.get_session("test-1")
        session.add_user_message("테스트")

        retrieved = conversation_store.get_session("test-1")
        assert len(retrieved.messages) == 1


class TestAIAssistant:
    """AI 어시스턴트 테스트"""

    @pytest.fixture
    def assistant(self):
        assistant = AIAssistant()
        assistant.compile(with_memory=False)
        return assistant

    def test_simple_chat(self, assistant):
        """간단한 대화 테스트"""
        response = assistant.chat("안녕하세요!", session_id="test")
        assert response is not None
        assert len(response) > 0

    def test_calculation_request(self, assistant):
        """계산 요청 테스트"""
        response = assistant.chat("1 + 1은 뭐야?", session_id="test-calc")
        assert "2" in response or "둘" in response

    def test_conversation_history(self, assistant):
        """대화 기록 테스트"""
        assistant.chat("제 이름은 테스터입니다.", session_id="test-history")
        history = assistant.get_conversation_history("test-history")

        assert len(history) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 마무리

### 학습한 내용

1. ✅ 도구 사용 에이전트 설계 및 구현
2. ✅ Human-in-the-Loop 패턴 적용
3. ✅ 스트리밍 응답 구현
4. ✅ 대화 메모리 관리
5. ✅ 웹 인터페이스 구현
6. ✅ LangSmith 통합

### 프로덕션 배포 체크리스트

- [ ] API 키 보안 (환경 변수, 시크릿 매니저)
- [ ] 레이트 리미팅 구현
- [ ] 에러 핸들링 강화
- [ ] 로깅 시스템 구축
- [ ] 모니터링 및 알림 설정
- [ ] 부하 테스트 수행
- [ ] 백업 및 복구 계획

### 추가 개선 아이디어

1. **음성 인터페이스**: Whisper API 통합
2. **이미지 처리**: GPT-4 Vision 활용
3. **플러그인 시스템**: 사용자 정의 도구 추가
4. **다국어 지원**: 자동 언어 감지 및 번역
5. **개인화**: 사용자 선호도 학습
