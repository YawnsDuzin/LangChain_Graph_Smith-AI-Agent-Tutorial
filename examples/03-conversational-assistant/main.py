"""
대화형 AI 어시스턴트 예제
========================
도구 사용 능력을 갖춘 대화형 AI 어시스턴트

사용법:
    python main.py
"""

import os
import math
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ============ 도구 정의 ============

@tool
def calculate(expression: str) -> str:
    """
    수학 표현식을 계산합니다.
    예: "2 + 2", "sqrt(16)", "10 * 5"
    """
    try:
        # 안전한 수학 함수들
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def get_current_time() -> str:
    """현재 날짜와 시간을 반환합니다."""
    now = datetime.now()
    return now.strftime("%Y년 %m월 %d일 %A %H:%M:%S")


@tool
def calculate_date(days: int, operation: str = "add") -> str:
    """
    오늘 날짜에서 일수를 더하거나 뺍니다.

    Args:
        days: 일 수
        operation: "add" (더하기) 또는 "subtract" (빼기)
    """
    today = datetime.now()
    if operation == "subtract":
        result = today - timedelta(days=days)
    else:
        result = today + timedelta(days=days)
    return f"결과: {result.strftime('%Y년 %m월 %d일 %A')}"


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """
    단위를 변환합니다.

    지원 단위:
    - 길이: km/mile, m/ft, cm/inch
    - 무게: kg/lb, g/oz
    - 온도: celsius/fahrenheit
    """
    conversions = {
        ("km", "mile"): lambda x: x * 0.621371,
        ("mile", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("cm", "inch"): lambda x: x * 0.393701,
        ("inch", "cm"): lambda x: x * 2.54,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("g", "oz"): lambda x: x * 0.035274,
        ("oz", "g"): lambda x: x * 28.3495,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"지원하지 않는 단위 변환: {from_unit} → {to_unit}"


TOOLS = [calculate, get_current_time, calculate_date, unit_convert]


# ============ 상태 정의 ============
class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]


# ============ 어시스턴트 클래스 ============
class ConversationalAssistant:
    SYSTEM_PROMPT = """당신은 유능하고 친절한 AI 어시스턴트입니다.

사용 가능한 도구:
- 🧮 calculate: 수학 계산
- 🕐 get_current_time: 현재 시간 조회
- 📅 calculate_date: 날짜 계산
- 📏 unit_convert: 단위 변환

규칙:
- 필요한 경우 도구를 사용하세요.
- 친근하지만 전문적으로 대답하세요.
- 한국어로 응답하세요."""

    def __init__(self, model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.app = None
        self._build_graph()

    def _assistant_node(self, state: AssistantState) -> dict:
        """어시스턴트 노드"""
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_use_tools(self, state: AssistantState) -> Literal["tools", "end"]:
        """도구 사용 여부 결정"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    def _build_graph(self):
        """그래프 구축"""
        graph = StateGraph(AssistantState)

        # 노드 추가
        graph.add_node("assistant", self._assistant_node)
        graph.add_node("tools", ToolNode(TOOLS))

        # 엣지 추가
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges(
            "assistant",
            self._should_use_tools,
            {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "assistant")

        checkpointer = MemorySaver()
        self.app = graph.compile(checkpointer=checkpointer)
        print("✅ 어시스턴트 그래프 컴파일 완료")

    def chat(self, message: str, session_id: str = "default") -> str:
        """채팅"""
        config = {"configurable": {"thread_id": session_id}}

        result = self.app.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        # 마지막 AI 응답 반환
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return "응답을 생성할 수 없습니다."


# ============ 메인 ============
def main():
    print("\n" + "="*50)
    print("🤖 대화형 AI 어시스턴트")
    print("="*50)
    print("\n사용 가능한 기능:")
    print("  🧮 '123 * 456 계산해줘'")
    print("  🕐 '지금 몇 시야?'")
    print("  📅 '오늘부터 100일 후는?'")
    print("  📏 '100km를 마일로 변환해줘'")
    print("\n명령어:")
    print("  /help - 도움말")
    print("  quit  - 종료")
    print("-"*50 + "\n")

    assistant = ConversationalAssistant()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input == "/help":
                print("""
📚 도움말

[계산]
• "2 + 2 계산해줘"
• "sqrt(144)는?"
• "100 * 50 / 25"

[시간/날짜]
• "지금 몇 시야?"
• "오늘 날짜 알려줘"
• "100일 후는 언제야?"
• "30일 전은?"

[단위 변환]
• "100km를 마일로"
• "68도 화씨를 섭씨로"
• "5kg를 파운드로"
""")
                continue

            if user_input.lower() in ["quit", "exit", "종료"]:
                print("👋 안녕히 가세요!")
                break

            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


if __name__ == "__main__":
    main()
