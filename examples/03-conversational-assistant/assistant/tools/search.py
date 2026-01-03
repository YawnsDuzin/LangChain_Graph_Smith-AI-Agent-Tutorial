# assistant/tools/search.py
"""
============================================================================
검색 도구 모듈
============================================================================

이 모듈은 웹 검색, 뉴스 검색, 위키피디아 검색 도구를 제공합니다.
LangChain의 @tool 데코레이터를 사용하여 LLM이 호출할 수 있는 도구로 정의합니다.

핵심 개념:
    - @tool 데코레이터: 함수를 LLM이 사용할 수 있는 도구로 변환
    - TavilySearchResults: AI 최적화된 웹 검색 API
    - WikipediaQueryRun: 위키피디아 검색 래퍼

사용 예시:
    from assistant.tools.search import web_search, news_search

    # 웹 검색
    result = web_search.invoke({"query": "파이썬 최신 버전"})

    # 뉴스 검색
    news = news_search.invoke({"topic": "AI", "days": 7})

============================================================================
"""
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional
import os


# ============================================================================
# @tool 데코레이터 상세 설명
# ============================================================================
# @tool은 LangChain에서 제공하는 핵심 데코레이터입니다.
# 일반 Python 함수를 LLM이 호출할 수 있는 도구(Tool)로 변환합니다.
#
# 동작 원리:
#   1. 함수의 이름, docstring, 타입 힌트를 분석
#   2. OpenAI Function Calling 형식의 스키마 생성
#   3. LLM에게 도구 정보 전달 시 이 스키마 사용
#
# 스키마 생성 규칙:
#   - 함수 이름 → 도구 이름
#   - docstring → 도구 설명 (LLM이 도구 선택 시 참고)
#   - 매개변수 타입 힌트 → JSON Schema 타입
#   - 매개변수 docstring → 매개변수 설명
#
# 예시 함수:
#   @tool
#   def add(a: int, b: int) -> int:
#       """두 숫자를 더합니다."""
#       return a + b
#
# 생성되는 스키마:
#   {
#       "name": "add",
#       "description": "두 숫자를 더합니다.",
#       "parameters": {
#           "type": "object",
#           "properties": {
#               "a": {"type": "integer"},
#               "b": {"type": "integer"}
#           },
#           "required": ["a", "b"]
#       }
#   }
#
# 주의사항:
#   1. docstring은 필수! LLM이 도구 용도를 파악하는 데 사용
#   2. 타입 힌트 필수! 스키마 생성에 필요
#   3. 명확한 매개변수 설명 → LLM이 올바른 값 전달
#
# 도구 호출 방법:
#   # 직접 호출
#   result = my_tool.invoke({"param1": "value1"})
#
#   # LLM을 통한 호출 (자동)
#   llm_with_tools = llm.bind_tools([my_tool])
#   response = llm_with_tools.invoke(messages)
#   # response.tool_calls에 호출 정보가 포함됨
# ============================================================================


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    ========================================================================
    인터넷에서 최신 정보를 검색합니다.
    ========================================================================

    이 도구는 Tavily 검색 API를 사용하여 웹을 검색합니다.
    Tavily는 AI 애플리케이션에 최적화된 검색 엔진으로,
    LLM이 처리하기 좋은 형태로 결과를 반환합니다.

    Args:
        query: 검색 쿼리
            예: "파이썬 3.12 새로운 기능", "오늘 주요 뉴스"
        max_results: 최대 결과 수 (기본값: 5)
            1~10 사이의 값 권장

    Returns:
        검색 결과 요약 (포맷팅된 문자열)
        각 결과는 제목, URL, 내용 스니펫 포함

    사용 예시:
        # 기본 검색
        result = web_search.invoke({"query": "LangChain 튜토리얼"})

        # 결과 수 제한
        result = web_search.invoke({
            "query": "Python 최신 뉴스",
            "max_results": 3
        })
    ========================================================================
    """
    try:
        # ====================================================================
        # TavilySearchResults 상세 설명
        # ====================================================================
        # TavilySearchResults는 LangChain Community에서 제공하는
        # Tavily 검색 API 래퍼입니다.
        #
        # Tavily API 특징:
        #   - AI 최적화: 검색 결과가 LLM 처리에 최적화
        #   - 실시간 검색: 최신 정보 접근 가능
        #   - 요약 기능: 검색 결과 자동 요약
        #   - 필터링: 도메인, 시간 범위 등 필터 지원
        #
        # 주요 매개변수:
        #   - max_results (int): 반환할 최대 결과 수
        #   - search_depth (str): "basic" 또는 "advanced"
        #   - include_domains (list): 포함할 도메인
        #   - exclude_domains (list): 제외할 도메인
        #
        # 반환값 구조:
        #   [
        #       {
        #           "title": "검색 결과 제목",
        #           "url": "https://example.com/page",
        #           "content": "검색 결과 내용 스니펫..."
        #       },
        #       ...
        #   ]
        #
        # API 키 설정:
        #   환경 변수 TAVILY_API_KEY에 API 키 설정 필요
        #   .env 파일: TAVILY_API_KEY=tvly-xxx...
        # ====================================================================
        search = TavilySearchResults(max_results=max_results)

        # invoke() 메서드로 검색 실행
        # 입력: 검색 쿼리 문자열
        # 출력: 검색 결과 리스트 (딕셔너리들)
        results = search.invoke(query)

        if not results:
            return "검색 결과가 없습니다."

        # 검색 결과 포맷팅
        # LLM과 사용자가 읽기 쉬운 형태로 변환
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No Title')
            url = result.get('url', 'N/A')
            # 내용이 너무 길면 300자로 제한
            content = result.get('content', '')[:300]
            formatted.append(f"[{i}] {title}\n    URL: {url}\n    {content}...")

        return "\n\n".join(formatted)

    except Exception as e:
        # 에러 발생 시 사용자 친화적 메시지 반환
        # LLM이 이 메시지를 보고 사용자에게 적절히 전달
        return f"검색 중 오류 발생: {str(e)}"


@tool
def news_search(topic: str, days: int = 7) -> str:
    """
    ========================================================================
    최근 뉴스를 검색합니다.
    ========================================================================

    특정 주제에 대한 최근 뉴스를 검색합니다.
    내부적으로 web_search 도구를 사용하여 뉴스 키워드로 검색합니다.

    Args:
        topic: 뉴스 주제
            예: "AI", "주식", "스포츠"
        days: 검색할 기간 (일 단위, 기본값: 7)
            최근 며칠 이내의 뉴스를 검색할지 지정

    Returns:
        관련 뉴스 목록

    사용 예시:
        # 최근 7일 AI 뉴스
        news = news_search.invoke({"topic": "인공지능"})

        # 최근 3일 경제 뉴스
        news = news_search.invoke({"topic": "경제", "days": 3})

    주의:
        이 도구는 web_search를 내부적으로 호출하므로
        Tavily API 키가 필요합니다.
    ========================================================================
    """
    # 뉴스 검색에 최적화된 쿼리 구성
    query = f"{topic} 뉴스 최근 {days}일"

    # 다른 도구를 호출할 때도 invoke() 사용
    # 딕셔너리로 매개변수 전달
    return web_search.invoke({"query": query, "max_results": 5})


@tool
def wikipedia_search(topic: str) -> str:
    """
    ========================================================================
    위키피디아에서 정보를 검색합니다.
    ========================================================================

    위키피디아에서 주제에 대한 정보를 검색하고 요약을 반환합니다.
    백과사전 스타일의 정확한 정보가 필요할 때 유용합니다.

    Args:
        topic: 검색할 주제
            예: "인공지능", "파이썬 프로그래밍 언어"

    Returns:
        위키피디아 정보 요약 (최대 2000자)

    사용 예시:
        # 기본 검색
        info = wikipedia_search.invoke({"topic": "머신러닝"})

    웹 검색 vs 위키피디아:
        - 웹 검색: 최신 정보, 다양한 출처
        - 위키피디아: 정확한 정의, 구조화된 정보
    ========================================================================
    """
    try:
        # ====================================================================
        # WikipediaQueryRun & WikipediaAPIWrapper 상세 설명
        # ====================================================================
        # LangChain Community에서 제공하는 위키피디아 검색 도구입니다.
        #
        # WikipediaAPIWrapper:
        #   위키피디아 API를 래핑하는 유틸리티 클래스입니다.
        #
        #   주요 매개변수:
        #   - lang (str): 언어 코드 (기본: "en", 한국어: "ko")
        #   - top_k_results (int): 반환할 최대 결과 수
        #   - doc_content_chars_max (int): 문서당 최대 문자 수
        #
        #   사용 예시:
        #       wrapper = WikipediaAPIWrapper(lang="ko", top_k_results=3)
        #
        # WikipediaQueryRun:
        #   WikipediaAPIWrapper를 사용하여 검색을 실행하는 도구입니다.
        #
        #   생성:
        #       tool = WikipediaQueryRun(api_wrapper=wrapper)
        #
        #   실행:
        #       result = tool.run("검색어")  # 문자열 반환
        #
        # 주의사항:
        #   - 인터넷 연결 필요
        #   - 위키피디아에 문서가 없으면 빈 결과 반환
        #   - 동음이의어 주의 (명확한 검색어 사용 권장)
        # ====================================================================
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        # API 래퍼 생성 (기본 설정 사용)
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        # 검색 실행 - run() 메서드 사용
        # WikipediaQueryRun은 .run() 메서드를 사용 (invoke가 아님)
        result = wikipedia.run(topic)

        # 결과가 너무 길면 2000자로 제한
        # LLM 컨텍스트 길이 제한 고려
        return result[:2000]

    except Exception as e:
        return f"위키피디아 검색 오류: {str(e)}"
