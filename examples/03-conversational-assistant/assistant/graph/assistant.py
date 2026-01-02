# assistant/graph/assistant.py
from typing import Literal, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langsmith import traceable

from assistant.graph.state import AssistantState, create_initial_state
from assistant.tools import ALL_TOOLS, TOOLS_REQUIRING_APPROVAL
from assistant.memory.conversation import conversation_store


class AIAssistant:
    """대화형 AI 어시스턴트"""

    SYSTEM_PROMPT = """당신은 유능하고 친절한 AI 어시스턴트입니다.

당신의 능력:
1. 🔍 인터넷 검색으로 최신 정보 제공
2. 🧮 수학 계산 및 단위 변환
3. 🐍 Python 코드 실행 및 설명
4. 📅 날짜/시간 계산
5. 📝 텍스트 요약

규칙:
- 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.
- 필요한 경우 도구를 사용하여 정보를 얻으세요.
- 모르는 것은 솔직히 모른다고 말하세요.
- 친근하지만 전문적인 톤을 유지하세요.
- 한국어로 응답하세요.

중요: 코드 실행이 필요한 경우, 사용자에게 먼저 확인을 받으세요."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self.model = model
        self.temperature = temperature
        self.tools = ALL_TOOLS
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True
        ).bind_tools(self.tools)

        self.graph = self._build_graph()
        self.app = None

    def _build_graph(self) -> StateGraph:
        """그래프 구축"""
        graph = StateGraph(AssistantState)

        # 노드 추가
        graph.add_node("process_input", self._process_input)
        graph.add_node("assistant", self._assistant_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("check_approval", self._check_approval)
        graph.add_node("generate_response", self._generate_response)

        # 엣지 추가
        graph.add_edge(START, "process_input")
        graph.add_edge("process_input", "assistant")

        # 어시스턴트 후 조건부 라우팅
        graph.add_conditional_edges(
            "assistant",
            self._route_after_assistant,
            {
                "tools": "tools",
                "check_approval": "check_approval",
                "respond": "generate_response"
            }
        )

        # 도구 실행 후 어시스턴트로 복귀
        graph.add_edge("tools", "assistant")

        # 승인 확인 후 라우팅
        graph.add_conditional_edges(
            "check_approval",
            self._route_after_approval,
            {
                "tools": "tools",
                "respond": "generate_response"
            }
        )

        graph.add_edge("generate_response", END)

        return graph

    @traceable(name="process_input")
    def _process_input(self, state: AssistantState) -> dict:
        """입력 처리"""
        user_input = state["user_input"]
        session_id = state["session_id"]

        # 대화 기록 가져오기
        memory = conversation_store.get_session(session_id)
        memory.add_user_message(user_input)

        return {
            "messages": [HumanMessage(content=user_input)]
        }

    @traceable(name="assistant")
    def _assistant_node(self, state: AssistantState) -> dict:
        """어시스턴트 노드"""
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]

        response = self.llm.invoke(messages)

        return {"messages": [response]}

    def _route_after_assistant(
        self,
        state: AssistantState
    ) -> Literal["tools", "check_approval", "respond"]:
        """어시스턴트 후 라우팅"""
        last_message = state["messages"][-1]

        # 도구 호출이 있는지 확인
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls

            # 승인이 필요한 도구가 있는지 확인
            for call in tool_calls:
                if call["name"] in TOOLS_REQUIRING_APPROVAL:
                    return "check_approval"

            return "tools"

        return "respond"

    @traceable(name="check_approval")
    def _check_approval(self, state: AssistantState) -> dict:
        """승인 확인 노드"""
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        pending_calls = []
        for call in tool_calls:
            pending_calls.append({
                "tool_name": call["name"],
                "tool_args": call["args"],
                "requires_approval": call["name"] in TOOLS_REQUIRING_APPROVAL,
                "approved": None
            })

        # 승인 대기 상태로 설정
        return {
            "pending_tool_calls": pending_calls,
            "requires_approval": True
        }

    def _route_after_approval(
        self,
        state: AssistantState
    ) -> Literal["tools", "respond"]:
        """승인 후 라우팅"""
        if state.get("user_approved"):
            return "tools"
        return "respond"

    @traceable(name="generate_response")
    def _generate_response(self, state: AssistantState) -> dict:
        """최종 응답 생성"""
        last_message = state["messages"][-1]

        if hasattr(last_message, "content"):
            response = last_message.content
        else:
            response = str(last_message)

        # 대화 기록에 추가
        session_id = state["session_id"]
        memory = conversation_store.get_session(session_id)
        memory.add_ai_message(response)

        return {"response": response}

    def compile(self, with_memory: bool = True):
        """그래프 컴파일"""
        if with_memory:
            checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=checkpointer)
        else:
            self.app = self.graph.compile()
        return self.app

    @traceable(name="chat")
    def chat(
        self,
        message: str,
        session_id: str = "default"
    ) -> str:
        """동기식 채팅"""
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(message, session_id)
        config = {"configurable": {"thread_id": session_id}}

        result = self.app.invoke(initial_state, config=config)

        return result["response"]

    async def achat(
        self,
        message: str,
        session_id: str = "default"
    ) -> str:
        """비동기식 채팅"""
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(message, session_id)
        config = {"configurable": {"thread_id": session_id}}

        result = await self.app.ainvoke(initial_state, config=config)

        return result["response"]

    def stream_chat(
        self,
        message: str,
        session_id: str = "default"
    ):
        """스트리밍 채팅"""
        if self.app is None:
            self.compile()

        initial_state = create_initial_state(message, session_id)
        config = {"configurable": {"thread_id": session_id}}

        for event in self.app.stream(
            initial_state,
            config=config,
            stream_mode="values"
        ):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    yield {
                        "type": "message",
                        "content": last_msg.content
                    }

                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    yield {
                        "type": "tool_call",
                        "calls": last_msg.tool_calls
                    }

    def approve_tool_execution(
        self,
        session_id: str,
        approved: bool = True
    ):
        """도구 실행 승인"""
        if self.app is None:
            return

        config = {"configurable": {"thread_id": session_id}}

        # 승인 상태 업데이트
        self.app.update_state(
            config,
            {"user_approved": approved, "requires_approval": False}
        )

    def get_conversation_history(
        self,
        session_id: str
    ) -> List[Dict]:
        """대화 기록 가져오기"""
        memory = conversation_store.get_session(session_id)
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in memory.messages
        ]

    def clear_conversation(self, session_id: str):
        """대화 기록 초기화"""
        memory = conversation_store.get_session(session_id)
        memory.clear()


# 사용 예시
if __name__ == "__main__":
    assistant = AIAssistant()

    print("AI 어시스턴트가 시작되었습니다. 'quit'로 종료합니다.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("안녕히 가세요!")
            break

        response = assistant.chat(user_input, session_id="cli-session")
        print(f"\nAssistant: {response}\n")
