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
