"""
멀티 에이전트 시스템 예제
========================
LangGraph를 활용한 연구-작성-검토 멀티 에이전트 협업 시스템

사용법:
    python main.py
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ============ 상태 정의 ============
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    research_result: str
    draft: str
    review_feedback: str
    next_agent: str
    iteration: int
    max_iterations: int
    is_approved: bool


# ============ 에이전트 정의 ============
class MultiAgentSystem:
    def __init__(self, model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.app = None
        self._build_graph()

    # ---- 연구원 에이전트 ----
    def researcher(self, state: AgentState) -> dict:
        """정보를 조사하는 연구원"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 철저한 연구 전문가입니다.
주어진 주제에 대해 깊이 있는 조사를 수행하고 핵심 정보를 정리합니다.

다음을 포함하세요:
- 주요 개념 설명
- 핵심 포인트 3-5개
- 관련 사례나 예시"""),
            ("human", "다음 주제를 조사해주세요: {task}")
        ])

        chain = prompt | self.llm
        result = chain.invoke({"task": state["task"]})

        return {
            "research_result": result.content,
            "messages": [AIMessage(content=f"[연구원] {result.content}", name="Researcher")],
            "next_agent": "supervisor"
        }

    # ---- 작성자 에이전트 ----
    def writer(self, state: AgentState) -> dict:
        """콘텐츠를 작성하는 작성자"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 전문 콘텐츠 작성자입니다.
연구 결과를 바탕으로 읽기 쉽고 잘 구조화된 글을 작성합니다.

작성 규칙:
- 명확한 제목 포함
- 논리적 구조 (서론, 본론, 결론)
- 핵심 내용 강조"""),
            ("human", """연구 결과를 바탕으로 글을 작성해주세요.

주제: {task}

연구 결과:
{research}

{feedback}""")
        ])

        feedback_text = ""
        if state.get("review_feedback"):
            feedback_text = f"\n\n이전 피드백 반영:\n{state['review_feedback']}"

        chain = prompt | self.llm
        result = chain.invoke({
            "task": state["task"],
            "research": state["research_result"],
            "feedback": feedback_text
        })

        return {
            "draft": result.content,
            "messages": [AIMessage(content=f"[작성자] {result.content[:500]}...", name="Writer")],
            "next_agent": "supervisor"
        }

    # ---- 검토자 에이전트 ----
    def reviewer(self, state: AgentState) -> dict:
        """콘텐츠를 검토하는 검토자"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 꼼꼼한 편집자입니다.
작성된 글을 검토하고 품질을 평가합니다.

평가 기준:
- 정확성 (연구 결과 반영)
- 구조 (논리적 흐름)
- 가독성 (이해하기 쉬움)

점수 (0-100)와 함께 피드백을 제공하세요.
80점 이상이면 승인, 미만이면 수정 요청을 하세요."""),
            ("human", """다음 글을 검토해주세요.

주제: {task}

글:
{draft}

점수와 피드백을 제공하세요. 마지막에 "승인" 또는 "수정필요"를 명시하세요.""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "task": state["task"],
            "draft": state["draft"]
        })

        is_approved = "승인" in result.content and "수정필요" not in result.content

        return {
            "review_feedback": result.content,
            "is_approved": is_approved,
            "messages": [AIMessage(content=f"[검토자] {result.content}", name="Reviewer")],
            "next_agent": "supervisor"
        }

    # ---- 슈퍼바이저 ----
    def supervisor(self, state: AgentState) -> dict:
        """워크플로우를 조율하는 슈퍼바이저"""
        research = state.get("research_result")
        draft = state.get("draft")
        is_approved = state.get("is_approved", False)
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)

        # 워크플로우 결정
        if not research:
            next_agent = "researcher"
        elif not draft:
            next_agent = "writer"
        elif is_approved or iteration >= max_iter:
            next_agent = "finish"
        elif state.get("review_feedback"):
            next_agent = "writer"  # 수정 필요
        else:
            next_agent = "reviewer"

        message = f"[슈퍼바이저] 다음 단계: {next_agent}"

        return {
            "next_agent": next_agent,
            "iteration": iteration + 1,
            "messages": [AIMessage(content=message, name="Supervisor")]
        }

    def _route(self, state: AgentState) -> str:
        """라우팅 함수"""
        return state.get("next_agent", "supervisor")

    def _build_graph(self):
        """그래프 구축"""
        graph = StateGraph(AgentState)

        # 노드 추가
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("researcher", self.researcher)
        graph.add_node("writer", self.writer)
        graph.add_node("reviewer", self.reviewer)

        # 시작
        graph.add_edge(START, "supervisor")

        # 슈퍼바이저에서 조건부 라우팅
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {
                "researcher": "researcher",
                "writer": "writer",
                "reviewer": "reviewer",
                "finish": END
            }
        )

        # 각 에이전트 후 슈퍼바이저로 복귀
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("reviewer", "supervisor")

        checkpointer = MemorySaver()
        self.app = graph.compile(checkpointer=checkpointer)
        print("✅ 멀티 에이전트 그래프 컴파일 완료")

    def run(self, task: str, session_id: str = "default") -> dict:
        """작업 실행"""
        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "task": task,
            "messages": [HumanMessage(content=task)],
            "research_result": "",
            "draft": "",
            "review_feedback": "",
            "next_agent": "supervisor",
            "iteration": 0,
            "max_iterations": 3,
            "is_approved": False
        }

        result = self.app.invoke(initial_state, config=config)

        return {
            "task": task,
            "draft": result.get("draft", ""),
            "approved": result.get("is_approved", False),
            "iterations": result.get("iteration", 0)
        }

    def stream_run(self, task: str, session_id: str = "default"):
        """스트리밍 실행"""
        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "task": task,
            "messages": [HumanMessage(content=task)],
            "research_result": "",
            "draft": "",
            "review_feedback": "",
            "next_agent": "supervisor",
            "iteration": 0,
            "max_iterations": 3,
            "is_approved": False
        }

        for event in self.app.stream(initial_state, config=config, stream_mode="updates"):
            yield event


# ============ 메인 ============
def main():
    print("\n" + "="*60)
    print("🤖 멀티 에이전트 콘텐츠 제작 시스템")
    print("="*60)
    print("\n에이전트 팀:")
    print("  📚 Researcher - 정보 조사")
    print("  ✍️  Writer    - 콘텐츠 작성")
    print("  ✅ Reviewer   - 품질 검토")
    print("  👔 Supervisor - 워크플로우 조율")
    print("\n종료하려면 'quit' 입력\n")

    system = MultiAgentSystem()

    while True:
        try:
            task = input("📝 작성할 주제: ").strip()

            if not task:
                continue

            if task.lower() in ["quit", "exit", "종료"]:
                print("👋 안녕히 가세요!")
                break

            print("\n🔄 멀티 에이전트가 작업을 시작합니다...\n")
            print("-"*50)

            # 스트리밍 실행
            for event in system.stream_run(task):
                for node, update in event.items():
                    if "messages" in update and update["messages"]:
                        msg = update["messages"][-1]
                        print(f"\n{msg.content}\n")

            print("-"*50)
            print("✅ 작업 완료!\n")

        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


if __name__ == "__main__":
    main()
