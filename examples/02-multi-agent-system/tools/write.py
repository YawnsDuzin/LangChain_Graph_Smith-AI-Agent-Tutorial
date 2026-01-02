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
