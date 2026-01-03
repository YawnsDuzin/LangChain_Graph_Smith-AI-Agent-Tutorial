# app/chains.py
"""
RAG 체인 모듈

이 모듈은 LangChain의 LCEL(LangChain Expression Language)을 사용하여
RAG 파이프라인의 체인을 구성합니다.

LCEL(LangChain Expression Language) 개념:
------------------------------------------
LCEL은 LangChain에서 체인을 구성하는 선언적 방식입니다.
파이프 연산자 `|`를 사용하여 컴포넌트를 연결합니다.

기본 구조:
    chain = component1 | component2 | component3

예시:
    chain = prompt | llm | parser
    # 1. prompt: 사용자 입력을 프롬프트로 변환
    # 2. llm: 프롬프트를 LLM에 전달
    # 3. parser: LLM 응답을 파싱

LCEL의 장점:
- 선언적이고 읽기 쉬운 코드
- 자동 배치, 스트리밍, 비동기 지원
- 쉬운 디버깅 및 트레이싱
- 컴포넌트 재사용 용이

RAG 파이프라인에서의 역할:
1. 문서 로딩 (Document Loading)
2. 청크 분할 (Text Splitting)
3. 임베딩 생성 (Embedding)
4. 벡터 저장 (Vector Store)
5. 검색 (Retrieval)
6. 생성 (Generation) <- 이 모듈
"""

from typing import List, Dict, Any

# =============================================================================
# LangChain 임포트 설명
# =============================================================================

# langchain_core.prompts.ChatPromptTemplate
# -------------------------------------------
# 대화형 모델용 프롬프트 템플릿입니다.
# 시스템 메시지, 사용자 메시지, AI 메시지를 구조화합니다.
#
# 프롬프트 구성요소:
# - system: AI의 역할/성격 정의
# - human: 사용자 입력
# - ai: AI 응답 (few-shot 예시용)
#
# 사용 예시:
#   prompt = ChatPromptTemplate.from_messages([
#       ("system", "당신은 친절한 AI입니다."),
#       ("human", "{question}")
#   ])
#   formatted = prompt.invoke({"question": "안녕?"})
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# langchain_core.output_parsers.StrOutputParser
# -----------------------------------------------
# LLM 출력을 문자열로 파싱합니다.
#
# LLM 출력 형식:
# - ChatModel: AIMessage 객체 반환
# - AIMessage.content: 실제 텍스트
#
# StrOutputParser:
# - AIMessage에서 content만 추출
# - 문자열로 반환
#
# 사용 예시:
#   chain = prompt | llm | StrOutputParser()
#   result = chain.invoke({"question": "질문"})  # 문자열 반환
from langchain_core.output_parsers import StrOutputParser

# langchain_core.runnables
# -------------------------
# LCEL의 핵심 컴포넌트들입니다.
#
# RunnablePassthrough:
# - 입력을 그대로 다음 단계로 전달
# - 변환 없이 통과 (passthrough)
# - 다른 Runnable과 병렬 실행 시 유용
#
# RunnableLambda:
# - 일반 Python 함수를 Runnable로 변환
# - 커스텀 처리 로직 삽입 가능
#
# 사용 예시:
#   # RunnablePassthrough
#   chain = {
#       "context": retriever,
#       "question": RunnablePassthrough()  # 입력을 그대로 전달
#   } | prompt | llm
#
#   # RunnableLambda
#   def format_docs(docs):
#       return "\n".join(doc.page_content for doc in docs)
#
#   chain = retriever | RunnableLambda(format_docs)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# langchain_openai.ChatOpenAI
# ----------------------------
# OpenAI의 Chat 모델 (GPT-4, GPT-3.5 등) 래퍼입니다.
#
# Chat 모델 vs Completion 모델:
# - Chat: 대화 형식 (메시지 리스트 입력)
# - Completion: 단일 텍스트 입력 (레거시)
#
# 주요 파라미터:
# - model: 사용할 모델 (gpt-4, gpt-4o, gpt-3.5-turbo 등)
# - temperature: 창의성 조절 (0.0~2.0)
# - max_tokens: 최대 출력 토큰 수
# - streaming: 스트리밍 활성화
#
# 사용 예시:
#   llm = ChatOpenAI(model="gpt-4", temperature=0.7)
#   response = llm.invoke([HumanMessage(content="안녕")])
from langchain_openai import ChatOpenAI

from langchain_core.documents import Document
from app.config import Config
from app.vectorstore import VectorStoreManager


class RAGChains:
    """
    RAG 관련 체인 클래스

    다양한 RAG 체인을 생성하고 관리합니다.

    제공하는 체인:
    1. 기본 RAG 체인: 단순 질문-응답
    2. 대화형 RAG 체인: 대화 기록 고려
    3. 질문 라우터: 질문 유형 분류

    사용 예시:
        chains = RAGChains()

        # 기본 RAG
        rag_chain = chains.create_rag_chain()
        result = rag_chain.invoke("LangChain이란?")

        # 대화형 RAG
        conv_chain = chains.create_conversational_rag_chain()
        result = conv_chain.invoke({
            "question": "더 자세히 알려줘",
            "chat_history": [...]
        })
    """

    def __init__(self):
        """
        RAGChains 초기화

        LLM과 리트리버를 설정합니다.
        """
        # =====================================================================
        # ChatOpenAI 상세 설명
        # =====================================================================
        #
        # 생성자 파라미터:
        #
        # model (str): 사용할 GPT 모델
        #   - "gpt-4": GPT-4, 가장 강력
        #   - "gpt-4-turbo": 더 빠르고 저렴
        #   - "gpt-4o": 멀티모달 지원
        #   - "gpt-4o-mini": 가장 빠르고 저렴
        #   - "gpt-3.5-turbo": 이전 세대
        #
        # temperature (float): 응답의 창의성 (0.0~2.0)
        #   - 0.0: 결정적, 항상 같은 응답
        #   - 0.7: 적당한 다양성 (기본값)
        #   - 1.5+: 매우 창의적
        #
        # max_tokens (int, optional): 최대 출력 토큰 수
        #   - 미지정 시 모델 기본값 사용
        #   - 비용 제어에 유용
        #
        # streaming (bool): 스트리밍 활성화
        #   - True: 토큰 단위로 실시간 출력
        #   - 대화형 UI에 유용
        #
        # openai_api_key (str, optional): API 키
        #   - 생략 시 환경변수 사용
        #
        # 주요 메서드:
        # - invoke(messages): 동기 호출
        # - ainvoke(messages): 비동기 호출
        # - stream(messages): 스트리밍 호출
        # - batch(message_lists): 배치 호출
        # =====================================================================
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,        # "gpt-4"
            temperature=Config.TEMPERATURE  # 0.7
        )

        # 벡터 스토어 매니저 및 리트리버 초기화
        self.vs_manager = VectorStoreManager()
        self.retriever = self.vs_manager.get_retriever()

    def _format_docs(self, docs: List[Document]) -> str:
        """
        문서들을 문자열로 포맷팅

        검색된 문서 청크들을 LLM이 이해하기 쉬운 형태로 변환합니다.

        Args:
            docs (List[Document]): 검색된 문서 리스트

        Returns:
            str: 포맷팅된 문자열

        포맷팅 형식:
            [문서 1] (출처: example.pdf)
            문서 내용...

            ---

            [문서 2] (출처: guide.md)
            문서 내용...
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def create_rag_chain(self):
        """
        기본 RAG 체인 생성

        단순한 질문-응답 RAG 체인을 생성합니다.
        대화 기록은 고려하지 않습니다.

        Returns:
            Runnable: 실행 가능한 RAG 체인

        체인 구조:
        1. 입력: 질문 문자열
        2. 리트리버로 관련 문서 검색
        3. 문서와 질문을 프롬프트에 포맷팅
        4. LLM으로 답변 생성
        5. 문자열로 파싱하여 반환

        사용 예시:
            chain = chains.create_rag_chain()
            answer = chain.invoke("LangChain의 주요 기능은?")
        """
        # =====================================================================
        # ChatPromptTemplate.from_messages() 상세 설명
        # =====================================================================
        #
        # 메시지 리스트로 프롬프트 템플릿을 생성합니다.
        #
        # 메시지 형식:
        # - ("system", "텍스트"): 시스템 메시지 (AI 역할 정의)
        # - ("human", "텍스트"): 사용자 메시지
        # - ("ai", "텍스트"): AI 메시지 (few-shot 예시)
        #
        # 변수 치환:
        # - {variable_name}: 실행 시 값으로 대체됨
        # - 예: {question} → "LangChain이란?"
        #
        # 다중 라인 텍스트:
        # - Python의 암묵적 문자열 연결 사용
        # - 또는 """ 삼중 따옴표 사용
        #
        # 예시:
        #   prompt = ChatPromptTemplate.from_messages([
        #       ("system", "당신은 AI입니다."),
        #       ("human", "{question}")
        #   ])
        #
        #   # 실행
        #   messages = prompt.invoke({"question": "안녕?"})
        #   # 결과: [SystemMessage, HumanMessage] 리스트
        # =====================================================================

        # RAG 프롬프트 정의
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

        # =====================================================================
        # LCEL 체인 구성 상세 설명
        # =====================================================================
        #
        # 체인 구조: dict | prompt | llm | parser
        #
        # 1단계: 입력 변환 (dict)
        # -------------------------
        # {
        #     "context": self.retriever | RunnableLambda(self._format_docs),
        #     "question": RunnablePassthrough()
        # }
        #
        # 동작:
        # - "context" 키:
        #   1. 입력을 retriever에 전달 → 문서 리스트 반환
        #   2. RunnableLambda로 _format_docs 함수 적용 → 문자열 변환
        # - "question" 키:
        #   RunnablePassthrough()로 입력을 그대로 전달
        #
        # 입력: "LangChain이란?"
        # 출력: {"context": "포맷된 문서들", "question": "LangChain이란?"}
        #
        # 2단계: 프롬프트 적용 (rag_prompt)
        # --------------------------------
        # 딕셔너리의 키-값을 템플릿 변수에 매핑
        # {context} → 포맷된 문서
        # {question} → 질문
        #
        # 3단계: LLM 호출 (self.llm)
        # ---------------------------
        # 프롬프트를 LLM에 전달하여 응답 생성
        # 반환: AIMessage 객체
        #
        # 4단계: 출력 파싱 (StrOutputParser())
        # ------------------------------------
        # AIMessage에서 content 추출
        # 반환: 문자열
        # =====================================================================

        chain = (
            {
                # context: 리트리버로 검색 후 포맷팅
                "context": self.retriever | RunnableLambda(self._format_docs),
                # question: 입력을 그대로 전달
                "question": RunnablePassthrough()
            }
            | rag_prompt    # 프롬프트 적용
            | self.llm      # LLM 호출
            | StrOutputParser()  # 문자열로 파싱
        )

        return chain

    def create_conversational_rag_chain(self):
        """
        대화형 RAG 체인 생성 (히스토리 포함)

        대화 기록을 고려하여 문맥을 유지하는 RAG 체인입니다.
        이전 대화를 참고하여 질문을 재작성하고 답변합니다.

        Returns:
            Runnable: 대화형 RAG 체인

        체인 구조:
        1. 입력: {"question": str, "chat_history": List[Message]}
        2. 대화 기록이 있으면 질문 재작성 (contextualization)
        3. 재작성된 질문으로 문서 검색
        4. 대화 기록 + 문서 + 질문으로 답변 생성

        질문 재작성 예시:
        - 이전 대화: "LangChain에 대해 알려줘" → AI 답변
        - 후속 질문: "설치 방법은?" → "LangChain 설치 방법은?"
        """
        # =====================================================================
        # MessagesPlaceholder 상세 설명
        # =====================================================================
        #
        # 동적 메시지 리스트를 프롬프트에 삽입합니다.
        # 대화 기록처럼 길이가 변하는 메시지에 유용합니다.
        #
        # 사용법:
        #   MessagesPlaceholder(variable_name="chat_history")
        #
        # 실행 시:
        #   prompt.invoke({
        #       "chat_history": [
        #           HumanMessage("이전 질문"),
        #           AIMessage("이전 답변"),
        #       ],
        #       "question": "후속 질문"
        #   })
        #
        # 결과:
        #   [SystemMessage, HumanMessage("이전 질문"),
        #    AIMessage("이전 답변"), HumanMessage("후속 질문")]
        # =====================================================================

        # 질문 재작성 프롬프트 (Contextualization)
        # 후속 질문을 독립적인 질문으로 변환
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """대화 기록을 참고하여 사용자의 최신 질문을 재작성하세요.
대화 기록 없이도 이해할 수 있는 독립적인 질문으로 만드세요.
질문을 재작성할 필요가 없으면 그대로 반환하세요."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # 질문 재작성 체인
        # 대화 기록을 보고 질문을 독립적으로 변환
        contextualize_chain = (
            contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        # 대화형 RAG 프롬프트
        # 대화 기록과 문서 컨텍스트를 모두 포함
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
            """
            컨텍스트 검색 함수

            대화 기록이 있으면 질문을 재작성하고,
            재작성된 질문으로 문서를 검색합니다.

            Args:
                inputs: {"question": str, "chat_history": List[Message]}

            Returns:
                검색된 컨텍스트와 원본 정보를 포함한 딕셔너리

            동작:
            1. 대화 기록 확인
            2. 기록이 있으면 질문 재작성
            3. 재작성된 질문으로 문서 검색
            4. 검색 결과 포맷팅
            """
            question = inputs["question"]
            chat_history = inputs.get("chat_history", [])

            # 대화 기록이 있으면 질문 재작성
            # 예: "그것의 설치 방법은?" → "LangChain의 설치 방법은?"
            if chat_history:
                standalone_question = contextualize_chain.invoke({
                    "question": question,
                    "chat_history": chat_history
                })
            else:
                standalone_question = question

            # =========================================================
            # retriever.invoke() 상세 설명
            # =========================================================
            #
            # 리트리버의 검색 메서드입니다.
            #
            # 입력: 검색 쿼리 (문자열)
            # 출력: List[Document]
            #
            # 내부 동작:
            # 1. 쿼리를 임베딩으로 변환
            # 2. 벡터 스토어에서 유사한 문서 검색
            # 3. Document 객체 리스트 반환
            #
            # 검색 결과 수는 as_retriever() 호출 시
            # search_kwargs={"k": 4}로 설정됨
            # =========================================================
            docs = self.retriever.invoke(standalone_question)
            context = self._format_docs(docs)

            return {
                "context": context,
                "question": question,
                "chat_history": chat_history,
                "retrieved_docs": docs  # 디버깅/출처 표시용
            }

        # =====================================================================
        # RunnablePassthrough.assign() 상세 설명
        # =====================================================================
        #
        # 입력 딕셔너리에 새 키-값 쌍을 추가합니다.
        # 기존 값은 유지하면서 새 값을 계산합니다.
        #
        # 사용법:
        #   RunnablePassthrough.assign(
        #       new_key=some_chain
        #   )
        #
        # 동작:
        # 입력: {"a": 1, "b": 2}
        # assign(c=lambda x: x["a"] + x["b"])
        # 출력: {"a": 1, "b": 2, "c": 3}
        #
        # 체인에서의 활용:
        #   chain = RunnablePassthrough.assign(
        #       answer=prompt | llm | parser
        #   )
        #
        # 입력 딕셔너리를 그대로 전달하면서
        # "answer" 키에 LLM 응답을 추가
        # =====================================================================

        # 대화형 RAG 체인 구성
        chain = (
            RunnableLambda(get_context)  # 컨텍스트 검색
            | {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "retrieved_docs": lambda x: x["retrieved_docs"]
            }
            | RunnablePassthrough.assign(
                # answer 키에 LLM 응답 추가
                answer=qa_prompt | self.llm | StrOutputParser()
            )
        )

        return chain

    def create_query_router(self):
        """
        질문 라우터 생성 - 문서 검색이 필요한지 판단

        질문이 문서 검색을 필요로 하는지 판단합니다.
        일반적인 대화인지 정보 검색인지 분류합니다.

        Returns:
            Runnable: 질문 라우터 체인

        분류 결과:
        - "search": 문서 검색 필요 (정보 질문)
        - "chat": 문서 검색 불필요 (인사, 감사 등)

        사용 예시:
            router = chains.create_query_router()

            result = router.invoke("LangChain이란?")  # "search"
            result = router.invoke("안녕하세요!")     # "chat"

        활용:
        - 불필요한 검색 비용 절감
        - 일반 대화에 더 자연스러운 응답 가능
        """
        # 라우터 프롬프트
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 질문 분류기입니다.
사용자의 질문이 문서 검색이 필요한 질문인지 판단하세요.

분류:
- "search": 특정 정보나 지식이 필요한 질문 (예: "LangChain이란?", "API 사용법을 알려줘")
- "chat": 일반적인 대화나 인사 (예: "안녕", "고마워", "잘 모르겠어")

오직 "search" 또는 "chat"만 응답하세요."""),
            ("human", "{question}")
        ])

        # 라우터 체인
        router_chain = (
            router_prompt
            | self.llm
            | StrOutputParser()
        )

        return router_chain


# =============================================================================
# 사용 예시 및 테스트
# =============================================================================
if __name__ == "__main__":
    """
    체인 테스트

    실행 방법:
        python -m app.chains
    """
    chains = RAGChains()

    print("=== 기본 RAG 체인 테스트 ===")
    rag_chain = chains.create_rag_chain()

    # 기본 RAG 테스트
    question = "LangChain의 주요 기능은?"
    print(f"질문: {question}")

    try:
        result = rag_chain.invoke(question)
        print(f"답변: {result}")
    except Exception as e:
        print(f"오류: {e}")

    print("\n=== 질문 라우터 테스트 ===")
    router = chains.create_query_router()

    test_questions = [
        "LangChain이 무엇인가요?",  # search
        "안녕하세요!",              # chat
        "API 사용법을 알려줘",      # search
        "고마워요",                 # chat
    ]

    for q in test_questions:
        try:
            result = router.invoke({"question": q})
            print(f"'{q}' → {result.strip()}")
        except Exception as e:
            print(f"오류: {e}")
