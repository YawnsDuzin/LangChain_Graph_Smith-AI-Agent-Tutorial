# assistant/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """애플리케이션 설정"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

    # Model Settings
    LLM_MODEL = "gpt-4"
    TEMPERATURE = 0.7

    # LangSmith
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-assistant-tutorial")

    @classmethod
    def validate(cls):
        """설정 검증"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        print("✅ 설정 검증 완료")
