# assistant/memory/conversation.py
from typing import List, Dict, Optional, Any
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_core.chat_history import BaseChatMessageHistory
from datetime import datetime
import json


class ConversationMemory(BaseChatMessageHistory):
    """대화 메모리 관리 클래스"""

    def __init__(
        self,
        session_id: str,
        max_messages: int = 50,
        max_tokens: int = 4000
    ):
        self.session_id = session_id
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: List[BaseMessage] = []
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "message_count": 0
        }

    @property
    def messages(self) -> List[BaseMessage]:
        """메시지 목록 반환"""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """메시지 추가"""
        self._messages.append(message)
        self._metadata["message_count"] += 1

        # 최대 메시지 수 초과 시 오래된 메시지 제거
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def add_user_message(self, message: str) -> None:
        """사용자 메시지 추가"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """AI 메시지 추가"""
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """대화 기록 초기화"""
        self._messages = []
        self._metadata["message_count"] = 0

    def get_recent_messages(self, n: int = 10) -> List[BaseMessage]:
        """최근 n개 메시지 반환"""
        return self._messages[-n:]

    def get_summary(self) -> str:
        """대화 요약 반환"""
        if not self._messages:
            return "대화 기록이 없습니다."

        summary = f"세션 ID: {self.session_id}\n"
        summary += f"총 메시지 수: {len(self._messages)}\n"
        summary += f"생성 시간: {self._metadata['created_at']}\n"

        return summary

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "messages": [
                {
                    "type": type(m).__name__,
                    "content": m.content
                }
                for m in self._messages
            ],
            "metadata": self._metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMemory":
        """딕셔너리에서 복원"""
        memory = cls(session_id=data["session_id"])
        memory._metadata = data.get("metadata", {})

        for msg_data in data.get("messages", []):
            msg_type = msg_data["type"]
            content = msg_data["content"]

            if msg_type == "HumanMessage":
                memory.add_message(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                memory.add_message(AIMessage(content=content))
            elif msg_type == "SystemMessage":
                memory.add_message(SystemMessage(content=content))

        return memory

    def save_to_file(self, filepath: str) -> None:
        """파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ConversationMemory":
        """파일에서 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ConversationStore:
    """여러 대화 세션 관리"""

    def __init__(self):
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_session(self, session_id: str) -> ConversationMemory:
        """세션 가져오기 (없으면 생성)"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """모든 세션 ID 목록"""
        return list(self._sessions.keys())

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """세션 요약"""
        if session_id in self._sessions:
            return self._sessions[session_id].get_summary()
        return None


# 전역 대화 저장소
conversation_store = ConversationStore()
