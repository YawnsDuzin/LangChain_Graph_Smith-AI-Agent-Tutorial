# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class Config:
    """애플리케이션 설정"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "documents"
    VECTORDB_DIR = BASE_DIR / "vectordb"

    # Model Settings
    LLM_MODEL = "gpt-4"
    EMBEDDING_MODEL = "text-embedding-3-small"
    TEMPERATURE = 0.7

    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 4

    # LangSmith
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-chatbot")

    @classmethod
    def validate(cls):
        """설정 검증"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        print("✅ 설정 검증 완료")
