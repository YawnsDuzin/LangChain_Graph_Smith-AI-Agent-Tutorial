# agents/researcher.py
from typing import Dict, Any
import re
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
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', content)
        return urls if urls else ["내부 분석 기반"]
