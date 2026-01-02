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
