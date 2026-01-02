# app/main.py
import os
from pathlib import Path
from app.config import Config
from app.document_loader import DocumentProcessor
from app.vectorstore import VectorStoreManager
from app.graph import RAGChatbot


def initialize_vectorstore(force_rebuild: bool = False):
    """벡터 스토어 초기화"""
    vs_manager = VectorStoreManager()

    # 기존 벡터 스토어 확인
    if Config.VECTORDB_DIR.exists() and not force_rebuild:
        try:
            vs_manager.load_vectorstore()
            print("✅ 기존 벡터 스토어 로드됨")
            return vs_manager
        except Exception as e:
            print(f"⚠️ 벡터 스토어 로드 실패: {e}")

    # 새로 생성
    print("🔄 벡터 스토어 새로 생성 중...")

    # 문서 처리
    processor = DocumentProcessor()
    chunks = processor.process_documents()

    if not chunks:
        print("⚠️ 처리할 문서가 없습니다.")
        print(f"   문서를 {Config.DATA_DIR} 디렉토리에 추가하세요.")
        return None

    # 벡터 스토어 생성
    vs_manager.create_vectorstore(chunks)

    return vs_manager


def run_interactive_chat():
    """대화형 챗봇 실행"""
    print("\n" + "="*50)
    print("📚 RAG 문서 Q&A 챗봇")
    print("="*50)
    print("질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("-"*50 + "\n")

    chatbot = RAGChatbot()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "종료"]:
                print("👋 안녕히 가세요!")
                break

            # 응답 생성
            response = chatbot.chat(user_input)

            print(f"\nBot: {response['answer']}")

            if response.get("sources"):
                print(f"📄 출처: {', '.join(response['sources'])}")

            print()

        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def main():
    """메인 함수"""
    print("🚀 RAG 챗봇 시작...")

    # 설정 검증
    Config.validate()

    # 벡터 스토어 초기화
    vs_manager = initialize_vectorstore()

    if vs_manager is None:
        # 샘플 문서 생성
        create_sample_documents()
        vs_manager = initialize_vectorstore(force_rebuild=True)

    # 대화형 챗봇 실행
    run_interactive_chat()


def create_sample_documents():
    """샘플 문서 생성"""
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    sample_content = """# LangChain 소개

LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 프레임워크입니다.

## 주요 기능

1. **모델 통합**: OpenAI, Anthropic 등 다양한 LLM을 지원합니다.
2. **체인**: 여러 컴포넌트를 연결하여 복잡한 워크플로우를 구성합니다.
3. **에이전트**: 도구를 사용하여 동적으로 작업을 수행합니다.
4. **메모리**: 대화 기록을 관리합니다.

## 설치 방법

```bash
pip install langchain langchain-openai
```

## 간단한 예시

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4")
response = chat.invoke("안녕하세요!")
print(response.content)
```

# LangGraph 소개

LangGraph는 LangChain 기반의 그래프 워크플로우 라이브러리입니다.

## 특징

- 상태 기반 워크플로우
- 순환 그래프 지원
- 체크포인팅 기능
- 멀티 에이전트 지원

# LangSmith 소개

LangSmith는 LLM 애플리케이션의 모니터링 및 평가 플랫폼입니다.

## 주요 기능

- 트레이싱: 모든 LLM 호출 추적
- 평가: 자동화된 품질 평가
- 프롬프트 관리: 버전 관리 및 공유
- 데이터셋: 테스트 데이터 관리
"""

    sample_file = Config.DATA_DIR / "langchain_guide.md"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_content)

    print(f"📄 샘플 문서 생성됨: {sample_file}")


if __name__ == "__main__":
    main()
