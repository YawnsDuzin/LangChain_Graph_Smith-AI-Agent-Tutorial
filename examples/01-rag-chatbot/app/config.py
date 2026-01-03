# app/config.py
"""
RAG 챗봇 설정 모듈

이 모듈은 애플리케이션 전반에서 사용되는 설정값들을 중앙 집중적으로 관리합니다.
환경 변수를 로드하고, 모델 설정, 경로 설정, RAG 파라미터 등을 정의합니다.

LangChain/LangSmith 관련 설정:
- OPENAI_API_KEY: OpenAI API 호출에 필요한 인증 키
- LANGCHAIN_API_KEY: LangSmith 추적 및 모니터링에 필요한 키
- LANGCHAIN_PROJECT: LangSmith에서 프로젝트를 구분하는 이름

환경 변수 설정 방법:
1. .env 파일 생성
2. 아래 내용 추가:
   OPENAI_API_KEY=your-openai-api-key
   LANGCHAIN_API_KEY=your-langsmith-api-key
   LANGCHAIN_PROJECT=rag-chatbot
   LANGCHAIN_TRACING_V2=true
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# =============================================================================
# load_dotenv() 함수 설명
# =============================================================================
# python-dotenv 라이브러리의 함수입니다.
# 프로젝트 루트의 .env 파일에서 환경 변수를 읽어 os.environ에 로드합니다.
#
# 작동 방식:
# 1. 현재 디렉토리에서 .env 파일을 찾습니다
# 2. 파일의 KEY=VALUE 쌍을 파싱합니다
# 3. 각 키-값을 os.environ에 설정합니다
#
# 사용 예시:
#   .env 파일:
#   OPENAI_API_KEY=sk-xxxxx
#
#   Python 코드:
#   load_dotenv()
#   key = os.getenv("OPENAI_API_KEY")  # "sk-xxxxx" 반환
# =============================================================================
load_dotenv()


class Config:
    """
    애플리케이션 설정 클래스

    이 클래스는 모든 설정값을 클래스 변수로 관리합니다.
    인스턴스를 생성하지 않고 Config.SETTING_NAME 형태로 접근합니다.

    사용 예시:
        from app.config import Config

        model = Config.LLM_MODEL  # "gpt-4" 반환
        Config.validate()  # 필수 설정 검증
    """

    # =========================================================================
    # API Keys (API 인증 키)
    # =========================================================================

    # OPENAI_API_KEY: OpenAI API 호출에 필요한 인증 키
    # - ChatOpenAI, OpenAIEmbeddings 등 OpenAI 관련 클래스에서 자동으로 사용됨
    # - https://platform.openai.com/api-keys 에서 발급 가능
    # - 형식: "sk-" 로 시작하는 문자열
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # LANGCHAIN_API_KEY: LangSmith 서비스 인증 키
    # - LangSmith에서 트레이싱, 평가, 모니터링에 사용됨
    # - https://smith.langchain.com 에서 발급 가능
    # - 이 키가 설정되고 LANGCHAIN_TRACING_V2=true면 자동으로 추적 활성화
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

    # =========================================================================
    # Paths (경로 설정)
    # =========================================================================

    # BASE_DIR: 프로젝트 루트 디렉토리
    # - __file__: 현재 파일(config.py)의 절대 경로
    # - .parent: 상위 디렉토리 (app/)
    # - .parent: 그 상위 디렉토리 (프로젝트 루트)
    BASE_DIR = Path(__file__).parent.parent

    # DATA_DIR: 문서 파일이 저장되는 디렉토리
    # - PDF, TXT, MD 파일을 이 디렉토리에 넣으면 RAG에서 사용
    DATA_DIR = BASE_DIR / "data" / "documents"

    # VECTORDB_DIR: ChromaDB 벡터 데이터베이스 저장 경로
    # - 임베딩된 문서가 영구 저장되는 위치
    # - 서버 재시작 후에도 데이터 유지
    VECTORDB_DIR = BASE_DIR / "vectordb"

    # =========================================================================
    # Model Settings (모델 설정)
    # =========================================================================

    # LLM_MODEL: 텍스트 생성에 사용할 OpenAI 모델
    #
    # 사용 가능한 모델들:
    # - "gpt-4": GPT-4, 가장 강력하지만 비용이 높음 ($30/1M input tokens)
    # - "gpt-4-turbo": GPT-4 Turbo, 더 빠르고 저렴 ($10/1M input tokens)
    # - "gpt-4o": GPT-4o, 멀티모달 지원 ($5/1M input tokens)
    # - "gpt-4o-mini": 가장 저렴하고 빠름 ($0.15/1M input tokens)
    # - "gpt-3.5-turbo": 이전 세대, 가장 저렴 ($0.5/1M input tokens)
    #
    # 선택 가이드:
    # - 품질 우선: gpt-4 또는 gpt-4-turbo
    # - 비용 우선: gpt-4o-mini 또는 gpt-3.5-turbo
    # - 균형: gpt-4o
    LLM_MODEL = "gpt-4"

    # EMBEDDING_MODEL: 텍스트를 벡터로 변환하는 임베딩 모델
    #
    # OpenAI 임베딩 모델:
    # - "text-embedding-3-small": 1536 차원, $0.02/1M tokens (권장)
    # - "text-embedding-3-large": 3072 차원, $0.13/1M tokens (고품질)
    # - "text-embedding-ada-002": 1536 차원, 이전 세대
    #
    # 임베딩이란?
    # - 텍스트를 숫자 벡터로 변환하는 과정
    # - 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치
    # - RAG에서 질문과 문서의 유사도를 계산하는 데 사용
    EMBEDDING_MODEL = "text-embedding-3-small"

    # TEMPERATURE: LLM 응답의 창의성/무작위성 조절 (0.0 ~ 2.0)
    #
    # 값에 따른 특성:
    # - 0.0: 가장 결정적, 항상 같은 응답 (사실 기반 Q&A에 적합)
    # - 0.3~0.5: 약간의 변화, 일관성 유지 (일반적인 챗봇)
    # - 0.7~0.9: 창의적, 다양한 응답 (스토리텔링, 브레인스토밍)
    # - 1.0+: 매우 창의적, 예측 불가 (시, 창작)
    #
    # RAG에서는 0.5~0.7 권장 (정확성 + 자연스러움)
    TEMPERATURE = 0.7

    # =========================================================================
    # RAG Settings (RAG 파이프라인 설정)
    # =========================================================================

    # CHUNK_SIZE: 문서를 분할할 때 각 청크의 최대 문자 수
    #
    # 청크 분할 이유:
    # 1. LLM 컨텍스트 윈도우 제한 (최대 입력 토큰 수)
    # 2. 관련 없는 내용 제외하여 정확도 향상
    # 3. 임베딩 품질 향상 (작은 단위가 의미 파악에 유리)
    #
    # 크기 선택 가이드:
    # - 500~800: 짧은 문서, 정밀한 검색
    # - 1000~1500: 일반적인 문서 (권장)
    # - 2000+: 긴 문서, 문맥 유지 중요
    #
    # 주의: 너무 작으면 문맥 손실, 너무 크면 관련 없는 내용 포함
    CHUNK_SIZE = 1000

    # CHUNK_OVERLAP: 인접 청크 간 겹치는 문자 수
    #
    # 오버랩이 필요한 이유:
    # - 문장이 청크 경계에서 잘리는 것 방지
    # - 문맥 연속성 유지
    # - 검색 정확도 향상
    #
    # 일반적으로 CHUNK_SIZE의 10~20% 설정
    # 예: CHUNK_SIZE=1000이면 OVERLAP=100~200
    CHUNK_OVERLAP = 200

    # RETRIEVAL_K: 검색 시 반환할 문서 청크 수
    #
    # 값에 따른 영향:
    # - 작은 값 (2~3): 가장 관련 높은 것만, 빠름, 적은 토큰 사용
    # - 중간 값 (4~6): 균형 (권장)
    # - 큰 값 (8~10): 더 많은 컨텍스트, 느림, 비용 증가
    #
    # 고려사항:
    # - LLM 컨텍스트 윈도우 크기
    # - 응답 지연 시간
    # - API 비용
    RETRIEVAL_K = 4

    # =========================================================================
    # LangSmith Settings (모니터링 설정)
    # =========================================================================

    # LANGCHAIN_PROJECT: LangSmith에서 프로젝트 식별 이름
    #
    # LangSmith 프로젝트란?
    # - 관련된 LLM 호출들을 그룹화하여 관리
    # - 대시보드에서 프로젝트별 메트릭 확인 가능
    # - 예: "rag-chatbot-dev", "rag-chatbot-prod"
    #
    # 환경변수 우선순위:
    # 1. LANGCHAIN_PROJECT 환경변수가 있으면 사용
    # 2. 없으면 기본값 "rag-chatbot" 사용
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-chatbot")

    @classmethod
    def validate(cls):
        """
        설정 검증 메서드

        필수 설정값들이 올바르게 설정되었는지 확인합니다.
        누락된 설정이 있으면 ValueError를 발생시킵니다.

        @classmethod 데코레이터:
        - 인스턴스 생성 없이 Config.validate() 형태로 호출 가능
        - cls 파라미터로 클래스 자체에 접근

        사용 예시:
            try:
                Config.validate()
                print("설정 완료!")
            except ValueError as e:
                print(f"설정 오류: {e}")
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 OPENAI_API_KEY=your-api-key 형식으로 추가하세요."
            )
        print("✅ 설정 검증 완료")
