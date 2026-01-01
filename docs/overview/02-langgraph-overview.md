# LangGraph 완벽 가이드

## 목차
1. [LangGraph란?](#langgraph란)
2. [핵심 개념](#핵심-개념)
3. [그래프 구조](#그래프-구조)
4. [상태 관리](#상태-관리)
5. [설치 및 설정](#설치-및-설정)
6. [기본 사용법](#기본-사용법)
7. [고급 패턴](#고급-패턴)
8. [멀티 에이전트 시스템](#멀티-에이전트-시스템)

---

## LangGraph란?

LangGraph는 **LLM 기반의 상태 기반(stateful) 멀티 액터 애플리케이션**을 구축하기 위한 라이브러리입니다. LangChain 팀에서 개발했으며, 복잡한 에이전트 워크플로우와 멀티 에이전트 시스템을 그래프 형태로 모델링할 수 있게 해줍니다.

### LangGraph vs LangChain

| 특성 | LangChain | LangGraph |
|------|-----------|-----------|
| 워크플로우 | 선형 체인 | 순환 그래프 |
| 상태 관리 | 제한적 | 풍부한 상태 관리 |
| 분기 | 단순 조건부 | 복잡한 조건부 라우팅 |
| 반복 | 어려움 | 자연스러운 루프 지원 |
| 에이전트 | 단일 에이전트 | 멀티 에이전트 협업 |
| 체크포인팅 | 없음 | 내장 지원 |

### 왜 LangGraph를 사용해야 하는가?

1. **순환 워크플로우**: 에이전트가 반복적으로 행동하고 피드백을 받을 수 있음
2. **상태 영속성**: 체크포인팅을 통해 상태를 저장하고 복원
3. **인간 개입**: 워크플로우 중간에 사람의 승인이나 입력을 받을 수 있음
4. **스트리밍**: 각 노드의 출력을 실시간으로 스트리밍
5. **디버깅**: LangSmith와 통합된 상세한 추적

---

## 핵심 개념

### 1. 상태(State)
그래프 전체에서 공유되는 데이터 구조입니다.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 메시지 목록 (add_messages는 메시지를 누적)
    messages: Annotated[list, add_messages]
    # 현재 단계
    current_step: str
    # 수집된 데이터
    collected_data: dict
```

### 2. 노드(Nodes)
그래프에서 실제 작업을 수행하는 함수들입니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4")

def chatbot_node(state: State) -> dict:
    """챗봇이 응답을 생성하는 노드"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def analyzer_node(state: State) -> dict:
    """사용자 의도를 분석하는 노드"""
    last_message = state["messages"][-1]
    # 분석 로직
    return {"current_step": "analyzed"}
```

### 3. 엣지(Edges)
노드 간의 연결을 정의합니다.

```python
from langgraph.graph import StateGraph, START, END

# 일반 엣지: 항상 다음 노드로 이동
graph.add_edge("node_a", "node_b")

# 조건부 엣지: 조건에 따라 다른 노드로 이동
def route_decision(state: State) -> str:
    if state["current_step"] == "complete":
        return "end"
    return "continue"

graph.add_conditional_edges(
    "decision_node",
    route_decision,
    {
        "end": END,
        "continue": "process_node"
    }
)
```

### 4. 체크포인터(Checkpointer)
상태를 저장하고 복원하는 메커니즘입니다.

```python
from langgraph.checkpoint.memory import MemorySaver

# 메모리 기반 체크포인터
checkpointer = MemorySaver()

# 그래프 컴파일 시 체크포인터 추가
app = graph.compile(checkpointer=checkpointer)
```

---

## 그래프 구조

### 기본 그래프 생성

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 1. 상태 정의
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int

# 2. 노드 함수 정의
def step_one(state: GraphState) -> dict:
    return {
        "messages": [("assistant", "1단계 완료")],
        "step_count": state.get("step_count", 0) + 1
    }

def step_two(state: GraphState) -> dict:
    return {
        "messages": [("assistant", "2단계 완료")],
        "step_count": state["step_count"] + 1
    }

# 3. 그래프 구성
graph = StateGraph(GraphState)

# 4. 노드 추가
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)

# 5. 엣지 추가
graph.add_edge(START, "step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", END)

# 6. 컴파일
app = graph.compile()

# 7. 실행
result = app.invoke({"messages": [], "step_count": 0})
```

### 그래프 시각화

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph 구조 예시                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                        ┌─────────┐                          │
│                        │  START  │                          │
│                        └────┬────┘                          │
│                             │                               │
│                             ▼                               │
│                      ┌─────────────┐                        │
│                      │   입력 처리  │                        │
│                      └──────┬──────┘                        │
│                             │                               │
│                             ▼                               │
│                      ┌─────────────┐                        │
│              ┌───────│  의도 분석   │───────┐               │
│              │       └─────────────┘       │               │
│              │                             │               │
│         질문 │                             │ 작업          │
│              ▼                             ▼               │
│       ┌─────────────┐             ┌─────────────┐         │
│       │   검색/RAG   │             │  도구 실행   │         │
│       └──────┬──────┘             └──────┬──────┘         │
│              │                           │                 │
│              └───────────┬───────────────┘                 │
│                          │                                 │
│                          ▼                                 │
│                   ┌─────────────┐                          │
│                   │   응답 생성  │                          │
│                   └──────┬──────┘                          │
│                          │                                 │
│                          ▼                                 │
│                   ┌─────────────┐                          │
│           ┌───────│  완료 확인   │───────┐                 │
│           │       └─────────────┘       │                 │
│           │                             │                 │
│      완료 │                             │ 계속            │
│           ▼                             │                 │
│       ┌───────┐                         │                 │
│       │  END  │            ┌────────────┘                 │
│       └───────┘            │                               │
│                            └──────────▶ (입력 처리로 돌아감) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 상태 관리

### 리듀서(Reducers)
상태 업데이트 방식을 정의합니다.

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph.message import add_messages

def merge_dicts(left: dict, right: dict) -> dict:
    """두 딕셔너리를 병합하는 커스텀 리듀서"""
    return {**left, **right}

class AdvancedState(TypedDict):
    # 메시지는 누적됨
    messages: Annotated[list, add_messages]

    # 숫자는 더해짐
    total_count: Annotated[int, add]

    # 딕셔너리는 병합됨
    metadata: Annotated[dict, merge_dicts]

    # 기본값 (마지막 값으로 덮어씀)
    current_status: str
```

### 상태 분기

```python
from langgraph.graph import StateGraph, START, END

def should_continue(state: AdvancedState) -> str:
    """상태에 따라 다음 노드 결정"""
    if state["total_count"] >= 5:
        return "finish"
    if "error" in state["metadata"]:
        return "error_handler"
    return "continue_processing"

graph = StateGraph(AdvancedState)

# 조건부 엣지 추가
graph.add_conditional_edges(
    "processor",
    should_continue,
    {
        "finish": END,
        "error_handler": "error_node",
        "continue_processing": "processor"  # 자기 자신으로 루프
    }
)
```

---

## 설치 및 설정

### 설치

```bash
# LangGraph 설치
pip install langgraph

# 관련 패키지
pip install langchain-openai langchain-community

# 체크포인팅을 위한 추가 패키지
pip install langgraph-checkpoint-sqlite  # SQLite 기반
pip install langgraph-checkpoint-postgres  # PostgreSQL 기반
```

### 환경 설정

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith 추적 활성화 (선택사항이지만 권장)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-tutorial"
```

---

## 기본 사용법

### 간단한 챗봇 그래프

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 상태 정의
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# 모델 초기화
model = ChatOpenAI(model="gpt-4", temperature=0.7)

# 챗봇 노드
def chatbot(state: ChatState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 그래프 구성
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 컴파일
app = graph.compile()

# 실행
result = app.invoke({
    "messages": [HumanMessage(content="안녕하세요!")]
})
print(result["messages"][-1].content)
```

### 도구를 사용하는 에이전트 그래프

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 도구 정의
@tool
def search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    return f"'{query}'에 대한 검색 결과입니다."

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다."""
    try:
        return str(eval(expression))
    except:
        return "계산 오류"

tools = [search, calculator]

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 모델 설정 (도구 바인딩)
model = ChatOpenAI(model="gpt-4").bind_tools(tools)

# 에이전트 노드
def agent(state: AgentState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 도구 사용 여부 확인
def should_use_tools(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# 그래프 구성
graph = StateGraph(AgentState)

# 노드 추가
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

# 엣지 추가
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_use_tools,
    {"tools": "tools", "end": END}
)
graph.add_edge("tools", "agent")  # 도구 실행 후 에이전트로 복귀

# 컴파일
app = graph.compile()

# 실행
result = app.invoke({
    "messages": [HumanMessage(content="123 * 456을 계산해주세요")]
})
```

### 스트리밍 실행

```python
# 노드별 스트리밍
for event in app.stream(
    {"messages": [HumanMessage(content="AI의 미래에 대해 알려주세요")]},
    stream_mode="values"
):
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 업데이트만 스트리밍
for event in app.stream(
    {"messages": [HumanMessage(content="AI의 미래에 대해 알려주세요")]},
    stream_mode="updates"
):
    for node, values in event.items():
        print(f"노드 '{node}' 업데이트:")
        print(values)
```

---

## 고급 패턴

### 1. Human-in-the-Loop

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_action: str
    approved: bool

def request_approval(state: ApprovalState) -> dict:
    """승인 요청"""
    return {
        "pending_action": "이메일 발송",
        "approved": False
    }

def execute_action(state: ApprovalState) -> dict:
    """승인된 작업 실행"""
    if state["approved"]:
        return {"messages": [("assistant", f"'{state['pending_action']}' 실행 완료")]}
    return {"messages": [("assistant", "작업이 취소되었습니다.")]}

def check_approval(state: ApprovalState) -> str:
    if state.get("approved"):
        return "execute"
    return "wait"

# 그래프 구성
graph = StateGraph(ApprovalState)
graph.add_node("request", request_approval)
graph.add_node("execute", execute_action)

graph.add_edge(START, "request")
graph.add_conditional_edges(
    "request",
    check_approval,
    {"execute": "execute", "wait": END}  # wait 상태에서 중단
)
graph.add_edge("execute", END)

# 체크포인터로 상태 저장
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 1단계: 승인 요청
config = {"configurable": {"thread_id": "approval-1"}}
result = app.invoke(
    {"messages": [], "approved": False},
    config=config
)

# 2단계: 승인 후 재개
result = app.invoke(
    {"approved": True},  # 승인 상태 업데이트
    config=config
)
```

### 2. 서브그래프

```python
from langgraph.graph import StateGraph, START, END

class SubState(TypedDict):
    value: int

class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    sub_result: int

# 서브그래프 정의
def create_sub_graph():
    def double(state: SubState) -> dict:
        return {"value": state["value"] * 2}

    def add_ten(state: SubState) -> dict:
        return {"value": state["value"] + 10}

    sub = StateGraph(SubState)
    sub.add_node("double", double)
    sub.add_node("add_ten", add_ten)
    sub.add_edge(START, "double")
    sub.add_edge("double", "add_ten")
    sub.add_edge("add_ten", END)

    return sub.compile()

sub_graph = create_sub_graph()

# 메인 그래프에서 서브그래프 사용
def use_subgraph(state: MainState) -> dict:
    sub_result = sub_graph.invoke({"value": 5})
    return {"sub_result": sub_result["value"]}  # 5 * 2 + 10 = 20

main_graph = StateGraph(MainState)
main_graph.add_node("process", use_subgraph)
main_graph.add_edge(START, "process")
main_graph.add_edge("process", END)

app = main_graph.compile()
```

### 3. 병렬 실행

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import asyncio

class ParallelState(TypedDict):
    input: str
    result_a: str
    result_b: str
    final_result: str

async def task_a(state: ParallelState) -> dict:
    await asyncio.sleep(1)  # 시뮬레이션
    return {"result_a": f"A 처리: {state['input']}"}

async def task_b(state: ParallelState) -> dict:
    await asyncio.sleep(1)
    return {"result_b": f"B 처리: {state['input']}"}

def combine_results(state: ParallelState) -> dict:
    return {
        "final_result": f"{state['result_a']} + {state['result_b']}"
    }

# 병렬 노드 그래프
graph = StateGraph(ParallelState)
graph.add_node("task_a", task_a)
graph.add_node("task_b", task_b)
graph.add_node("combine", combine_results)

# 병렬 분기
graph.add_edge(START, "task_a")
graph.add_edge(START, "task_b")

# 병합
graph.add_edge("task_a", "combine")
graph.add_edge("task_b", "combine")
graph.add_edge("combine", END)

app = graph.compile()

# 비동기 실행
async def main():
    result = await app.ainvoke({"input": "테스트"})
    print(result["final_result"])

asyncio.run(main())
```

---

## 멀티 에이전트 시스템

### 슈퍼바이저 패턴

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

# 에이전트 정의
def researcher(state: MultiAgentState) -> dict:
    """연구 에이전트"""
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke([
        SystemMessage(content="당신은 연구 전문가입니다. 주어진 주제에 대해 조사합니다."),
        *state["messages"]
    ])
    return {"messages": [response]}

def writer(state: MultiAgentState) -> dict:
    """작성 에이전트"""
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke([
        SystemMessage(content="당신은 작가입니다. 연구 결과를 바탕으로 글을 작성합니다."),
        *state["messages"]
    ])
    return {"messages": [response]}

def reviewer(state: MultiAgentState) -> dict:
    """검토 에이전트"""
    model = ChatOpenAI(model="gpt-4")
    response = model.invoke([
        SystemMessage(content="당신은 편집자입니다. 작성된 글을 검토하고 피드백을 제공합니다."),
        *state["messages"]
    ])
    return {"messages": [response]}

def supervisor(state: MultiAgentState) -> dict:
    """슈퍼바이저: 다음 에이전트 결정"""
    model = ChatOpenAI(model="gpt-4")

    response = model.invoke([
        SystemMessage(content="""당신은 팀 슈퍼바이저입니다.
        현재 대화를 보고 다음에 어떤 에이전트가 작업해야 할지 결정하세요.
        가능한 선택: researcher, writer, reviewer, FINISH
        오직 에이전트 이름만 응답하세요."""),
        *state["messages"]
    ])

    next_agent = response.content.strip().lower()
    return {"next_agent": next_agent}

def route_to_agent(state: MultiAgentState) -> str:
    next_agent = state.get("next_agent", "").lower()
    if next_agent == "finish" or next_agent not in ["researcher", "writer", "reviewer"]:
        return "end"
    return next_agent

# 그래프 구성
graph = StateGraph(MultiAgentState)

# 노드 추가
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

# 엣지 추가
graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "end": END
    }
)

# 각 에이전트 후 슈퍼바이저로 복귀
for agent in ["researcher", "writer", "reviewer"]:
    graph.add_edge(agent, "supervisor")

# 컴파일
app = graph.compile()

# 실행
result = app.invoke({
    "messages": [HumanMessage(content="AI 기술의 미래에 대한 블로그 글을 작성해주세요.")]
})
```

### 에이전트 간 협업 패턴

```
┌─────────────────────────────────────────────────────────────────┐
│               멀티 에이전트 협업 아키텍처                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌──────────────┐                           │
│                      │  사용자 입력  │                           │
│                      └──────┬───────┘                           │
│                             │                                   │
│                             ▼                                   │
│                     ┌───────────────┐                           │
│                     │  슈퍼바이저    │◄────────────────┐        │
│                     │  (Supervisor) │                 │        │
│                     └───────┬───────┘                 │        │
│                             │                         │        │
│            ┌────────────────┼────────────────┐       │        │
│            │                │                │       │        │
│            ▼                ▼                ▼       │        │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│        │
│    │   연구 에이전트 ││   작성 에이전트 ││   검토 에이전트 ││        │
│    │  (Researcher) ││    (Writer)   ││  (Reviewer)  ││        │
│    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘│        │
│           │                │                │       │        │
│           └────────────────┴────────────────┴───────┘        │
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │                    공유 상태                         │     │
│    │  • messages: 대화 기록                              │     │
│    │  • research_data: 연구 결과                         │     │
│    │  • draft: 초안                                      │     │
│    │  • feedback: 피드백                                 │     │
│    └─────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 체크포인팅과 지속성

### SQLite 체크포인터

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite 체크포인터 생성
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-session-1"}}

    # 첫 번째 상호작용
    result1 = app.invoke(
        {"messages": [HumanMessage(content="안녕하세요")]},
        config=config
    )

    # 대화 계속 (이전 상태 유지)
    result2 = app.invoke(
        {"messages": [HumanMessage(content="아까 뭐라고 했죠?")]},
        config=config
    )
```

### 상태 복원

```python
# 특정 체크포인트로 복원
checkpoint_id = "checkpoint-abc123"
config = {
    "configurable": {
        "thread_id": "user-session-1",
        "checkpoint_id": checkpoint_id
    }
}

# 해당 시점의 상태에서 계속
result = app.invoke({"messages": [HumanMessage(content="계속하기")]}, config=config)
```

---

## 다음 단계

- [LangSmith 가이드](./03-langsmith-overview.md): 모니터링 및 디버깅
- [튜토리얼: 멀티 에이전트 시스템](../tutorials/02-multi-agent-system.md)
- [튜토리얼: 대화형 AI 어시스턴트](../tutorials/03-conversational-assistant.md)
