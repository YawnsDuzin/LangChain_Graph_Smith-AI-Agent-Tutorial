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
