# agents/base.py
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable
import os


class BaseAgent(ABC):
    """기본 에이전트 클래스"""

    def __init__(
        self,
        name: str,
        role: str,
        tools: List = None,
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self.name = name
        self.role = role
        self.tools = tools or []
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """에이전트의 시스템 프롬프트"""
        pass

    def create_prompt(self) -> ChatPromptTemplate:
        """프롬프트 템플릿 생성"""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ])

    @traceable(name="agent_invoke")
    def invoke(self, state: Dict) -> Dict:
        """에이전트 실행"""
        messages = state.get("messages", [])
        task = state.get("task", "")

        prompt = self.create_prompt()
        chain = prompt | self.llm

        response = chain.invoke({
            "messages": messages,
            "input": task
        })

        return self.process_response(response, state)

    @abstractmethod
    def process_response(self, response: Any, state: Dict) -> Dict:
        """응답 처리"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, role={self.role})"
