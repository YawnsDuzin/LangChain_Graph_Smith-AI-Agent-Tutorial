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
