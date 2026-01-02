# tests/test_rag.py
import pytest
from langsmith import Client, evaluate
from app.graph import RAGChatbot
from app.chains import RAGChains


class TestRAGChatbot:
    """RAG 챗봇 테스트"""

    @pytest.fixture
    def chatbot(self):
        return RAGChatbot()

    def test_general_chat(self, chatbot):
        """일반 대화 테스트"""
        response = chatbot.chat("안녕하세요!")
        assert "answer" in response
        assert response["query_type"] == "chat"

    def test_rag_query(self, chatbot):
        """RAG 쿼리 테스트"""
        response = chatbot.chat("LangChain이 무엇인가요?")
        assert "answer" in response
        assert response["query_type"] == "search"

    def test_conversation_memory(self, chatbot):
        """대화 메모리 테스트"""
        chatbot.set_thread("test-thread")

        # 첫 번째 질문
        response1 = chatbot.chat("제 이름은 홍길동입니다.")

        # 두 번째 질문 (기억 테스트)
        response2 = chatbot.chat("제 이름이 뭐라고 했죠?")
        assert "홍길동" in response2["answer"]


def create_evaluation_dataset():
    """평가 데이터셋 생성"""
    client = Client()

    dataset = client.create_dataset(
        dataset_name="rag-chatbot-eval",
        description="RAG 챗봇 평가 데이터셋"
    )

    examples = [
        {
            "inputs": {"question": "LangChain이란?"},
            "outputs": {"answer": "LLM 기반 애플리케이션 개발 프레임워크"}
        },
        {
            "inputs": {"question": "LangGraph의 특징은?"},
            "outputs": {"answer": "상태 기반 그래프 워크플로우"}
        },
        {
            "inputs": {"question": "LangSmith의 주요 기능은?"},
            "outputs": {"answer": "트레이싱, 평가, 모니터링"}
        }
    ]

    for example in examples:
        client.create_example(
            inputs=example["inputs"],
            outputs=example["outputs"],
            dataset_id=dataset.id
        )

    return dataset


def run_evaluation():
    """평가 실행"""
    chatbot = RAGChatbot()

    def predict(inputs: dict) -> dict:
        response = chatbot.chat(inputs["question"])
        return {"answer": response["answer"]}

    def relevance_evaluator(run, example) -> dict:
        """관련성 평가"""
        from langchain_openai import ChatOpenAI

        judge = ChatOpenAI(model="gpt-4", temperature=0)
        question = example.inputs["question"]
        answer = run.outputs["answer"]
        reference = example.outputs["answer"]

        prompt = f"""
질문: {question}
정답 키워드: {reference}
생성된 답변: {answer}

생성된 답변이 질문에 관련성이 있고 정답 키워드를 포함하는지 평가하세요.
0~1 사이의 점수만 응답하세요.
"""
        response = judge.invoke(prompt)
        try:
            score = float(response.content.strip())
        except:
            score = 0.5

        return {"key": "relevance", "score": score}

    results = evaluate(
        predict,
        data="rag-chatbot-eval",
        evaluators=[relevance_evaluator],
        experiment_prefix="rag-eval"
    )

    return results


if __name__ == "__main__":
    # 평가 데이터셋 생성
    # create_evaluation_dataset()

    # 평가 실행
    results = run_evaluation()
    print(f"평가 완료: {results}")
