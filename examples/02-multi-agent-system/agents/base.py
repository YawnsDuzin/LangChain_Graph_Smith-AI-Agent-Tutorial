# agents/base.py
"""
기본 에이전트 클래스 모듈

이 모듈은 멀티 에이전트 시스템의 기반이 되는 추상 에이전트 클래스를 정의합니다.
모든 구체적 에이전트(Researcher, Writer, Reviewer 등)는 이 클래스를 상속합니다.

에이전트 패턴 개념:
------------------
LLM 에이전트는 다음 요소로 구성됩니다:
1. LLM (두뇌): 추론과 결정을 담당
2. 도구 (손): 외부 시스템과 상호작용
3. 프롬프트 (성격): 에이전트의 역할과 행동 정의
4. 메모리 (기억): 대화 기록과 상태 유지

멀티 에이전트 시스템:
- 여러 전문 에이전트가 협력하여 복잡한 작업 수행
- 각 에이전트는 특정 역할에 특화
- Supervisor 패턴: 중앙 에이전트가 작업 분배
- Hierarchical 패턴: 계층적 책임 구조

이 모듈에서 사용되는 LangChain 개념:
- ChatOpenAI: LLM 래퍼
- bind_tools: 도구 바인딩
- ChatPromptTemplate: 프롬프트 템플릿
- @traceable: LangSmith 추적 데코레이터
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict

# =============================================================================
# LangChain 임포트 설명
# =============================================================================

# langchain_openai.ChatOpenAI
# ----------------------------
# OpenAI의 Chat 모델을 사용하기 위한 래퍼 클래스입니다.
#
# 주요 특징:
# - 다양한 GPT 모델 지원 (gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo)
# - 도구 바인딩 지원 (function calling)
# - 스트리밍, 비동기 호출 지원
#
# 도구 바인딩 (Tool Binding):
# - bind_tools(tools): 모델에 도구를 연결
# - 모델이 도구 호출 여부와 인자를 결정
# - OpenAI Function Calling 기능 활용
from langchain_openai import ChatOpenAI

# langchain_core.messages
# ------------------------
# LLM 대화에서 사용되는 메시지 타입들입니다.
#
# 메시지 유형:
# - HumanMessage: 사용자 입력
# - AIMessage: AI 응답
# - SystemMessage: 시스템 지시사항
# - ToolMessage: 도구 실행 결과
#
# 메시지 구조:
#   message.content: 텍스트 내용
#   message.type: 메시지 유형
#   message.tool_calls: 도구 호출 정보 (AIMessage)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# langchain_core.prompts
# -----------------------
# 프롬프트 템플릿 클래스들입니다.
#
# ChatPromptTemplate:
# - 대화형 모델용 프롬프트 생성
# - system, human, ai 메시지 조합
#
# MessagesPlaceholder:
# - 동적 메시지 리스트 삽입
# - 대화 기록 등 가변 길이 데이터에 유용
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# langsmith.traceable
# --------------------
# LangSmith 추적 데코레이터입니다.
#
# @traceable 데코레이터를 함수에 적용하면:
# - 함수 호출이 LangSmith에 기록됨
# - 입력, 출력, 실행 시간 추적
# - 중첩된 호출도 계층적으로 표시
#
# 사용법:
#   @traceable
#   def my_function(arg):
#       return result
#
#   @traceable(name="custom_name", run_type="chain")
#   def another_function(arg):
#       return result
#
# run_type 옵션:
# - "chain": 체인 실행
# - "tool": 도구 실행
# - "llm": LLM 호출
# - "prompt": 프롬프트 처리
from langsmith import traceable

import os


class BaseAgent(ABC):
    """
    기본 에이전트 추상 클래스

    모든 에이전트의 기반이 되는 추상 클래스입니다.
    구체적 에이전트는 이 클래스를 상속하여 구현합니다.

    추상 클래스(ABC) 개념:
    - 직접 인스턴스화 불가
    - @abstractmethod 메서드는 하위 클래스에서 반드시 구현
    - 공통 기능과 인터페이스 정의

    Attributes:
        name (str): 에이전트 이름
        role (str): 에이전트 역할 설명
        tools (List): 사용 가능한 도구 리스트
        llm (ChatOpenAI): LLM 인스턴스

    사용 예시:
        class MyAgent(BaseAgent):
            @property
            def system_prompt(self) -> str:
                return "당신은 전문 에이전트입니다."

            def process_response(self, response, state):
                return {"result": response.content}

        agent = MyAgent(name="my_agent", role="custom")
        result = agent.invoke(state)
    """

    def __init__(
        self,
        name: str,
        role: str,
        tools: List = None,
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        """
        BaseAgent 초기화

        Args:
            name (str): 에이전트 식별 이름
            role (str): 에이전트의 역할/책임 설명
            tools (List, optional): 에이전트가 사용할 도구 리스트
            model (str): 사용할 OpenAI 모델
            temperature (float): LLM 응답의 창의성 (0.0~2.0)

        LLM 초기화 상세 설명:
        ---------------------
        ChatOpenAI(model, temperature)
        - model: GPT 모델 버전
        - temperature: 출력 다양성 조절
          - 0.0: 결정적 (항상 같은 응답)
          - 0.7: 적당한 다양성
          - 1.0+: 높은 창의성

        도구 바인딩 (bind_tools):
        -------------------------
        llm.bind_tools(tools)

        도구를 LLM에 바인딩하면:
        1. 모델이 도구 스키마를 인식
        2. 필요시 도구 호출 결정
        3. AIMessage.tool_calls에 호출 정보 포함

        도구 정의 형식:
        @tool
        def my_tool(arg1: str, arg2: int) -> str:
            '''도구 설명'''
            return result

        bind_tools 후 응답 구조:
        response.content: 텍스트 응답 (도구 호출 시 비어있을 수 있음)
        response.tool_calls: [
            {"name": "tool_name", "args": {...}, "id": "..."}
        ]
        """
        self.name = name
        self.role = role
        self.tools = tools or []

        # LLM 초기화
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # 도구가 있으면 바인딩
        # =====================================================================
        # bind_tools() 상세 설명
        # =====================================================================
        # bind_tools(tools, tool_choice=None)
        #
        # 파라미터:
        # - tools: 도구 리스트 (@tool 데코레이터로 정의된 함수들)
        # - tool_choice: 도구 선택 강제 옵션
        #   - None: 모델이 자유롭게 결정
        #   - "auto": 모델이 자동 결정 (기본값)
        #   - "required": 반드시 도구 사용
        #   - {"type": "function", "function": {"name": "tool_name"}}: 특정 도구 강제
        #
        # 내부 동작:
        # 1. 각 도구의 JSON 스키마 생성
        # 2. OpenAI API의 functions/tools 파라미터로 전달
        # 3. 모델이 적절한 도구와 인자 결정
        #
        # 반환값:
        # - 도구가 바인딩된 새 LLM 인스턴스
        # - 원본 LLM은 변경되지 않음
        # =====================================================================
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        에이전트의 시스템 프롬프트

        각 에이전트의 역할, 성격, 지시사항을 정의합니다.
        하위 클래스에서 반드시 구현해야 합니다.

        @property 데코레이터:
        - 메서드를 속성처럼 접근 가능하게 함
        - agent.system_prompt 형태로 호출 (괄호 없음)

        @abstractmethod 데코레이터:
        - 하위 클래스에서 반드시 구현해야 함
        - 구현하지 않으면 인스턴스화 시 에러

        Returns:
            str: 시스템 프롬프트 문자열

        구현 예시:
            @property
            def system_prompt(self) -> str:
                return '''당신은 연구 전문가입니다.
                다음 규칙을 따르세요:
                1. 정확한 정보만 제공
                2. 출처 명시
                3. 객관적 관점 유지'''
        """
        pass

    def create_prompt(self) -> ChatPromptTemplate:
        """
        프롬프트 템플릿 생성

        시스템 프롬프트, 대화 기록, 현재 입력을 조합한 템플릿을 생성합니다.

        Returns:
            ChatPromptTemplate: 생성된 프롬프트 템플릿

        프롬프트 구조:
        1. system: 에이전트의 역할/성격 정의
        2. MessagesPlaceholder: 이전 대화 기록 삽입
        3. human: 현재 작업/입력

        ChatPromptTemplate.from_messages() 상세:
        -----------------------------------------
        메시지 튜플 리스트로 템플릿 생성

        형식: (role, content)
        - "system": 시스템 메시지
        - "human": 사용자 메시지
        - "ai": AI 응답

        MessagesPlaceholder:
        - variable_name: 대화 기록을 받을 변수명
        - optional: True면 비어있어도 허용 (기본 False)

        사용 예시:
            prompt = agent.create_prompt()
            messages = prompt.invoke({
                "messages": [HumanMessage("이전 질문"), AIMessage("이전 답변")],
                "input": "현재 작업"
            })
        """
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),           # 시스템 프롬프트
            MessagesPlaceholder(variable_name="messages"),  # 대화 기록
            ("human", "{input}")                      # 현재 입력
        ])

    @traceable(name="agent_invoke")
    def invoke(self, state: Dict) -> Dict:
        """
        에이전트 실행

        상태를 입력받아 LLM을 호출하고 결과를 반환합니다.

        Args:
            state (Dict): 현재 워크플로우 상태
                - messages: 대화 기록
                - task: 수행할 작업

        Returns:
            Dict: 업데이트된 상태

        @traceable 데코레이터:
        ----------------------
        이 메서드 호출이 LangSmith에 기록됩니다.
        - name="agent_invoke": LangSmith에 표시될 이름
        - 입력 상태와 반환값이 추적됨
        - 하위 LLM 호출도 중첩 표시

        LCEL 체인 (prompt | llm):
        -------------------------
        파이프 연산자로 컴포넌트 연결

        chain = prompt | llm 의 동작:
        1. prompt.invoke(inputs) → 메시지 리스트 생성
        2. llm.invoke(messages) → AI 응답 생성

        chain.invoke({...}):
        - 딕셔너리를 입력으로 받아
        - 체인 전체 실행 후 결과 반환
        """
        # 상태에서 데이터 추출
        messages = state.get("messages", [])
        task = state.get("task", "")

        # 프롬프트 템플릿 생성
        prompt = self.create_prompt()

        # =====================================================================
        # LCEL 체인 구성: prompt | llm
        # =====================================================================
        # 파이프 연산자(|)로 컴포넌트 연결
        #
        # 실행 흐름:
        # 1. prompt.invoke({"messages": [...], "input": task})
        #    → ChatPromptValue (메시지 리스트)
        # 2. llm.invoke(messages)
        #    → AIMessage (응답)
        #
        # 체인의 장점:
        # - 선언적 코드
        # - 자동 타입 변환
        # - 스트리밍, 비동기 지원
        # =====================================================================
        chain = prompt | self.llm

        # 체인 실행
        response = chain.invoke({
            "messages": messages,
            "input": task
        })

        # 응답 처리 (하위 클래스에서 구현)
        return self.process_response(response, state)

    @abstractmethod
    def process_response(self, response: Any, state: Dict) -> Dict:
        """
        응답 처리 추상 메서드

        LLM 응답을 처리하여 상태 업데이트를 반환합니다.
        하위 클래스에서 반드시 구현해야 합니다.

        Args:
            response: LLM 응답 (AIMessage)
            state (Dict): 현재 상태

        Returns:
            Dict: 업데이트할 상태 필드들

        구현 예시:
            def process_response(self, response, state):
                # 응답 파싱
                content = response.content

                # 도구 호출 확인
                if response.tool_calls:
                    tool_result = self.execute_tools(response.tool_calls)
                    return {"tool_results": tool_result}

                # 일반 응답 처리
                return {
                    "messages": [response],
                    "agent_output": content
                }
        """
        pass

    def __repr__(self):
        """문자열 표현"""
        return f"{self.__class__.__name__}(name={self.name}, role={self.role})"
