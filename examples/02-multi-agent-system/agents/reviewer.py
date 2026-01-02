# agents/reviewer.py
from typing import Dict, Any
import re
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
