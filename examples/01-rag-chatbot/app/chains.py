# app/chains.py
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from app.config import Config
from app.vectorstore import VectorStoreManager


class RAGChains:
    """RAG 관련 체인 클래스"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.vs_manager = VectorStoreManager()
        self.retriever = self.vs_manager.get_retriever()

    def _format_docs(self, docs: List[Document]) -> str:
        """문서들을 문자열로 포맷팅"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def create_rag_chain(self):
        """기본 RAG 체인 생성"""

        # RAG 프롬프트
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 문서 기반 Q&A 어시스턴트입니다.
주어진 컨텍스트를 바탕으로 질문에 정확하게 답변해주세요.

규칙:
1. 컨텍스트에 있는 정보만 사용하세요.
2. 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
3. 가능하면 출처를 언급하세요.
4. 답변은 명확하고 간결하게 작성하세요."""),
            ("human", """컨텍스트:
{context}

질문: {question}

답변:""")
        ])

        # 체인 구성
        chain = (
            {
                "context": self.retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def create_conversational_rag_chain(self):
        """대화형 RAG 체인 생성 (히스토리 포함)"""

        # 질문 재작성 프롬프트
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """대화 기록을 참고하여 사용자의 최신 질문을 재작성하세요.
대화 기록 없이도 이해할 수 있는 독립적인 질문으로 만드세요.
질문을 재작성할 필요가 없으면 그대로 반환하세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # 질문 재작성 체인
        contextualize_chain = (
            contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        # 대화형 RAG 프롬프트
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 친절한 문서 기반 Q&A 어시스턴트입니다.
주어진 컨텍스트와 대화 기록을 바탕으로 답변해주세요.

규칙:
1. 컨텍스트에 있는 정보를 우선 사용하세요.
2. 이전 대화 내용을 참고하여 자연스럽게 답변하세요.
3. 정보가 없으면 모른다고 솔직히 말하세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """컨텍스트:
{context}

질문: {question}

답변:""")
        ])

        def get_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """컨텍스트 검색"""
            question = inputs["question"]
            chat_history = inputs.get("chat_history", [])

            # 대화 기록이 있으면 질문 재작성
            if chat_history:
                standalone_question = contextualize_chain.invoke({
                    "question": question,
                    "chat_history": chat_history
                })
            else:
                standalone_question = question

            # 문서 검색
            docs = self.retriever.invoke(standalone_question)
            context = self._format_docs(docs)

            return {
                "context": context,
                "question": question,
                "chat_history": chat_history,
                "retrieved_docs": docs
            }

        # 대화형 RAG 체인
        chain = (
            RunnableLambda(get_context)
            | {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "retrieved_docs": lambda x: x["retrieved_docs"]
            }
            | RunnablePassthrough.assign(
                answer=qa_prompt | self.llm | StrOutputParser()
            )
        )

        return chain

    def create_query_router(self):
        """질문 라우터 생성 - 문서 검색이 필요한지 판단"""

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 질문 분류기입니다.
사용자의 질문이 문서 검색이 필요한 질문인지 판단하세요.

분류:
- "search": 특정 정보나 지식이 필요한 질문 (예: "LangChain이란?", "API 사용법을 알려줘")
- "chat": 일반적인 대화나 인사 (예: "안녕", "고마워", "잘 모르겠어")

오직 "search" 또는 "chat"만 응답하세요."""),
            ("human", "{question}")
        ])

        router_chain = (
            router_prompt
            | self.llm
            | StrOutputParser()
        )

        return router_chain


# 사용 예시
if __name__ == "__main__":
    chains = RAGChains()

    # 기본 RAG 테스트
    rag_chain = chains.create_rag_chain()
    result = rag_chain.invoke("LangChain의 주요 기능은?")
    print(f"답변: {result}")
