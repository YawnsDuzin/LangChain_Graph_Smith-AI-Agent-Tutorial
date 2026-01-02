# agents/supervisor.py
from typing import Dict, Any, Literal, List
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
