# tests/test_assistant.py
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.graph.assistant import AIAssistant
from assistant.tools import calculate, get_current_time
from assistant.memory.conversation import ConversationMemory, conversation_store


class TestTools:
    """도구 테스트"""

    def test_calculate(self):
        """계산 도구 테스트"""
        result = calculate.invoke({"expression": "2 + 2"})
        assert "4" in result

    def test_get_current_time(self):
        """시간 도구 테스트"""
        result = get_current_time.invoke({})
        assert "년" in result
        assert "월" in result


class TestConversationMemory:
    """대화 메모리 테스트"""

    def test_add_messages(self):
        """메시지 추가 테스트"""
        memory = ConversationMemory("test-session")
        memory.add_user_message("안녕하세요")
        memory.add_ai_message("안녕하세요! 무엇을 도와드릴까요?")

        assert len(memory.messages) == 2

    def test_clear_messages(self):
        """메시지 초기화 테스트"""
        memory = ConversationMemory("test-session")
        memory.add_user_message("테스트")
        memory.clear()

        assert len(memory.messages) == 0

    def test_conversation_store(self):
        """대화 저장소 테스트"""
        session = conversation_store.get_session("test-1")
        session.add_user_message("테스트")

        retrieved = conversation_store.get_session("test-1")
        assert len(retrieved.messages) == 1


class TestAIAssistant:
    """AI 어시스턴트 테스트"""

    @pytest.fixture
    def assistant(self):
        assistant = AIAssistant()
        assistant.compile(with_memory=False)
        return assistant

    def test_simple_chat(self, assistant):
        """간단한 대화 테스트"""
        response = assistant.chat("안녕하세요!", session_id="test")
        assert response is not None
        assert len(response) > 0

    def test_calculation_request(self, assistant):
        """계산 요청 테스트"""
        response = assistant.chat("1 + 1은 뭐야?", session_id="test-calc")
        assert "2" in response or "둘" in response

    def test_conversation_history(self, assistant):
        """대화 기록 테스트"""
        assistant.chat("제 이름은 테스터입니다.", session_id="test-history")
        history = assistant.get_conversation_history("test-history")

        assert len(history) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
