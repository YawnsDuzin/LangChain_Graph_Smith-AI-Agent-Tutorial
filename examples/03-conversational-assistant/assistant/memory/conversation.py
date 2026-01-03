# assistant/memory/conversation.py
"""
============================================================================
대화 메모리 모듈
============================================================================

이 모듈은 대화 세션의 메모리를 관리합니다.
LangChain의 BaseChatMessageHistory를 확장하여 커스텀 메모리 시스템을 구현합니다.

핵심 개념:
    - BaseChatMessageHistory: LangChain의 채팅 기록 인터페이스
    - 세션 관리: 여러 사용자/대화 세션 관리
    - 메시지 타입: HumanMessage, AIMessage, SystemMessage
    - 직렬화: JSON으로 대화 저장/로드

메모리 전략:
    - 최대 메시지 수 제한: 오래된 메시지 자동 삭제
    - 세션 기반: session_id로 대화 분리
    - 인메모리: 빠른 접근, 프로세스 종료 시 손실

프로덕션 고려사항:
    - Redis, SQLite 등 영구 저장소 사용 권장
    - LangChain의 RedisChatMessageHistory, SQLChatMessageHistory 활용

사용 예시:
    from assistant.memory.conversation import conversation_store

    # 세션 가져오기
    memory = conversation_store.get_session("user-123")

    # 메시지 추가
    memory.add_user_message("안녕하세요!")
    memory.add_ai_message("안녕하세요! 무엇을 도와드릴까요?")

    # 기록 조회
    for msg in memory.messages:
        print(msg.content)

============================================================================
"""
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


# ============================================================================
# 대화 메모리 클래스
# ============================================================================
class ConversationMemory(BaseChatMessageHistory):
    """
    ========================================================================
    대화 메모리 관리 클래스
    ========================================================================

    BaseChatMessageHistory 상세 설명:
    --------------------------------------------------------------------------
    BaseChatMessageHistory는 LangChain에서 정의한 추상 베이스 클래스(ABC)입니다.
    채팅 메시지 기록을 관리하는 표준 인터페이스를 제공합니다.

    필수 구현 메서드:
        - messages (property): 저장된 메시지 리스트 반환
        - add_message(message): 메시지 추가
        - clear(): 모든 메시지 삭제

    LangChain에서 제공하는 구현체들:
        - ChatMessageHistory: 기본 인메모리 구현
        - RedisChatMessageHistory: Redis 기반
        - SQLChatMessageHistory: SQL 데이터베이스 기반
        - FileChatMessageHistory: 파일 기반

    커스텀 구현의 장점:
        - 추가 메타데이터 저장
        - 커스텀 직렬화 로직
        - 특수한 만료 정책

    사용처:
        - RunnableWithMessageHistory에서 사용
        - 대화 체인의 메모리 관리
        - LangGraph 상태와 별도로 대화 이력 유지

    Attributes:
        session_id: 세션 식별자
        max_messages: 최대 저장 메시지 수
        max_tokens: 최대 토큰 수 (미구현)
    ========================================================================
    """

    def __init__(
        self,
        session_id: str,
        max_messages: int = 50,
        max_tokens: int = 4000
    ):
        """
        ====================================================================
        ConversationMemory 초기화
        ====================================================================

        Args:
            session_id: 대화 세션 식별자
                같은 ID를 사용하면 대화가 이어집니다.
            max_messages: 최대 저장 메시지 수 (기본값: 50)
                초과 시 오래된 메시지부터 삭제
            max_tokens: 최대 토큰 수 (기본값: 4000)
                현재 미구현, 향후 토큰 기반 제한에 사용

        초기화 항목:
            - _messages: 메시지 저장 리스트
            - _metadata: 세션 메타데이터 (생성 시간 등)
        ====================================================================
        """
        self.session_id = session_id
        self.max_messages = max_messages
        self.max_tokens = max_tokens

        # 메시지 저장 리스트 (private)
        # 언더스코어 prefix는 Python에서 private 변수 관례
        self._messages: List[BaseMessage] = []

        # 메타데이터 저장
        # ====================================================================
        # datetime.now().isoformat():
        #   현재 시간을 ISO 8601 형식 문자열로 변환
        #   예: "2024-01-15T14:30:00.123456"
        #
        # ISO 8601 형식의 장점:
        #   - 국제 표준
        #   - 문자열 정렬이 시간순 정렬과 일치
        #   - JSON 직렬화에 적합
        # ====================================================================
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "message_count": 0
        }

    # ========================================================================
    # BaseChatMessageHistory 필수 구현: messages property
    # ========================================================================
    @property
    def messages(self) -> List[BaseMessage]:
        """
        ====================================================================
        메시지 목록 반환
        ====================================================================

        @property 데코레이터:
        --------------------------------------------------------------------
        메서드를 속성처럼 접근할 수 있게 합니다.

        사용법:
            # property 없이
            msgs = memory.get_messages()

            # property 사용
            msgs = memory.messages  # 괄호 없이 접근

        장점:
            - 더 직관적인 API
            - getter/setter 로직 캡슐화
            - 계산된 값을 속성처럼 제공

        Returns:
            List[BaseMessage]: 저장된 메시지 리스트
        ====================================================================
        """
        return self._messages

    # ========================================================================
    # BaseChatMessageHistory 필수 구현: add_message
    # ========================================================================
    def add_message(self, message: BaseMessage) -> None:
        """
        ====================================================================
        메시지 추가
        ====================================================================

        메시지를 기록에 추가하고, 최대 개수를 초과하면 오래된 메시지를 삭제합니다.

        Args:
            message: 추가할 메시지 (BaseMessage 또는 하위 클래스)

        메시지 타입:
        --------------------------------------------------------------------
        BaseMessage의 주요 하위 클래스:

            HumanMessage:
                사용자 입력 메시지
                생성: HumanMessage(content="안녕하세요")

            AIMessage:
                AI 응답 메시지
                생성: AIMessage(content="안녕하세요!")
                tool_calls 속성으로 도구 호출 정보 포함 가능

            SystemMessage:
                시스템 지시 메시지 (AI 역할 정의 등)
                생성: SystemMessage(content="당신은 도움이 되는 AI입니다")

            ToolMessage:
                도구 실행 결과
                생성: ToolMessage(content="결과", tool_call_id="...")

        메시지 공통 속성:
            - content: 메시지 내용 (str 또는 list)
            - additional_kwargs: 추가 데이터 (dict)
            - id: 메시지 고유 ID (optional)
        ====================================================================
        """
        self._messages.append(message)
        self._metadata["message_count"] += 1

        # 최대 메시지 수 초과 시 오래된 메시지 제거
        # ====================================================================
        # 슬라이싱을 사용한 리스트 제한:
        #
        # self._messages[-max_messages:] 동작:
        #   - 리스트의 마지막 max_messages개 요소만 유지
        #   - 오래된(앞쪽) 메시지가 제거됨
        #
        # 예시:
        #   messages = [1, 2, 3, 4, 5]
        #   max = 3
        #   messages[-3:]  # [3, 4, 5]
        # ====================================================================
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def add_user_message(self, message: str) -> None:
        """
        ====================================================================
        사용자 메시지 추가 (편의 메서드)
        ====================================================================

        문자열을 HumanMessage로 래핑하여 추가합니다.

        Args:
            message: 사용자 메시지 내용 (문자열)

        사용 예시:
            memory.add_user_message("안녕하세요!")
            # 내부적으로 HumanMessage(content="안녕하세요!") 생성
        ====================================================================
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """
        ====================================================================
        AI 메시지 추가 (편의 메서드)
        ====================================================================

        문자열을 AIMessage로 래핑하여 추가합니다.

        Args:
            message: AI 응답 내용 (문자열)

        사용 예시:
            memory.add_ai_message("안녕하세요! 무엇을 도와드릴까요?")
        ====================================================================
        """
        self.add_message(AIMessage(content=message))

    # ========================================================================
    # BaseChatMessageHistory 필수 구현: clear
    # ========================================================================
    def clear(self) -> None:
        """
        ====================================================================
        대화 기록 초기화
        ====================================================================

        모든 메시지를 삭제하고 카운터를 리셋합니다.
        세션 ID와 생성 시간은 유지됩니다.
        ====================================================================
        """
        self._messages = []
        self._metadata["message_count"] = 0

    def get_recent_messages(self, n: int = 10) -> List[BaseMessage]:
        """
        ====================================================================
        최근 n개 메시지 반환
        ====================================================================

        Args:
            n: 반환할 메시지 수 (기본값: 10)

        Returns:
            최근 n개의 메시지 리스트

        사용 예시:
            # 최근 5개 메시지
            recent = memory.get_recent_messages(5)
        ====================================================================
        """
        return self._messages[-n:]

    def get_summary(self) -> str:
        """
        ====================================================================
        대화 요약 반환
        ====================================================================

        세션의 기본 정보를 문자열로 반환합니다.

        Returns:
            세션 ID, 메시지 수, 생성 시간 정보
        ====================================================================
        """
        if not self._messages:
            return "대화 기록이 없습니다."

        summary = f"세션 ID: {self.session_id}\n"
        summary += f"총 메시지 수: {len(self._messages)}\n"
        summary += f"생성 시간: {self._metadata['created_at']}\n"

        return summary

    def to_dict(self) -> Dict:
        """
        ====================================================================
        딕셔너리로 변환 (직렬화)
        ====================================================================

        대화 내용을 JSON 직렬화 가능한 딕셔너리로 변환합니다.
        파일 저장이나 API 응답에 사용됩니다.

        Returns:
            Dict: 직렬화된 대화 데이터
                - session_id: 세션 ID
                - messages: 메시지 리스트 (type과 content)
                - metadata: 메타데이터

        직렬화 전략:
        --------------------------------------------------------------------
        BaseMessage 객체는 직접 JSON으로 변환할 수 없으므로,
        각 메시지를 {"type": "클래스명", "content": "내용"} 형태로 변환합니다.

        type(m).__name__:
            객체의 클래스 이름을 문자열로 반환
            예: HumanMessage → "HumanMessage"
        ====================================================================
        """
        return {
            "session_id": self.session_id,
            "messages": [
                {
                    "type": type(m).__name__,  # 클래스 이름
                    "content": m.content
                }
                for m in self._messages
            ],
            "metadata": self._metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMemory":
        """
        ====================================================================
        딕셔너리에서 복원 (역직렬화)
        ====================================================================

        @classmethod 데코레이터:
        --------------------------------------------------------------------
        클래스 메서드를 정의합니다.
        인스턴스가 아닌 클래스 자체를 첫 번째 인자(cls)로 받습니다.

        팩토리 메서드 패턴:
            저장된 데이터에서 객체를 생성하는 패턴입니다.
            __init__ 대신 별도의 생성 메서드를 제공합니다.

        Args:
            data: to_dict()로 생성된 딕셔너리

        Returns:
            ConversationMemory: 복원된 메모리 인스턴스

        사용 예시:
            # 저장
            data = memory.to_dict()
            json_str = json.dumps(data)

            # 복원
            data = json.loads(json_str)
            memory = ConversationMemory.from_dict(data)
        ====================================================================
        """
        # 새 인스턴스 생성
        memory = cls(session_id=data["session_id"])
        memory._metadata = data.get("metadata", {})

        # 메시지 복원
        for msg_data in data.get("messages", []):
            msg_type = msg_data["type"]
            content = msg_data["content"]

            # 타입에 따라 적절한 메시지 클래스 사용
            if msg_type == "HumanMessage":
                memory.add_message(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                memory.add_message(AIMessage(content=content))
            elif msg_type == "SystemMessage":
                memory.add_message(SystemMessage(content=content))

        return memory

    def save_to_file(self, filepath: str) -> None:
        """
        ====================================================================
        파일로 저장
        ====================================================================

        대화 내용을 JSON 파일로 저장합니다.

        Args:
            filepath: 저장할 파일 경로

        파일 저장 과정:
        --------------------------------------------------------------------
        1. to_dict()로 직렬화
        2. json.dump()로 파일에 쓰기

        json.dump() 매개변수:
            - ensure_ascii=False: 유니코드 그대로 저장 (한글 등)
            - indent=2: 들여쓰기로 가독성 향상

        사용 예시:
            memory.save_to_file("conversations/user-123.json")
        ====================================================================
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ConversationMemory":
        """
        ====================================================================
        파일에서 로드
        ====================================================================

        JSON 파일에서 대화 내용을 복원합니다.

        Args:
            filepath: 로드할 파일 경로

        Returns:
            ConversationMemory: 복원된 메모리 인스턴스

        사용 예시:
            memory = ConversationMemory.load_from_file(
                "conversations/user-123.json"
            )
        ====================================================================
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# 대화 세션 저장소
# ============================================================================
class ConversationStore:
    """
    ========================================================================
    여러 대화 세션 관리
    ========================================================================

    여러 사용자/세션의 대화를 관리하는 저장소입니다.
    session_id를 키로 사용하여 각 세션의 메모리를 관리합니다.

    싱글톤 패턴:
    --------------------------------------------------------------------------
    이 클래스의 인스턴스(conversation_store)를 모듈 레벨에서 생성하여
    전역적으로 공유합니다. 이렇게 하면:
        - 모든 모듈에서 같은 저장소 사용
        - 세션 데이터 일관성 유지
        - 명시적인 전달 없이 접근 가능

    사용 예시:
        from assistant.memory.conversation import conversation_store

        # 세션 가져오기 (없으면 생성)
        memory = conversation_store.get_session("user-123")

        # 세션 목록
        sessions = conversation_store.list_sessions()

        # 세션 삭제
        conversation_store.delete_session("user-123")
    ========================================================================
    """

    def __init__(self):
        """
        ====================================================================
        ConversationStore 초기화
        ====================================================================

        _sessions 딕셔너리:
            키: session_id (str)
            값: ConversationMemory 인스턴스
        ====================================================================
        """
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_session(self, session_id: str) -> ConversationMemory:
        """
        ====================================================================
        세션 가져오기 (없으면 생성)
        ====================================================================

        Lazy Initialization 패턴:
        --------------------------------------------------------------------
        세션이 처음 요청될 때 생성합니다.
        미리 모든 세션을 생성하지 않아 메모리 효율적입니다.

        Args:
            session_id: 세션 식별자

        Returns:
            ConversationMemory: 해당 세션의 메모리
        ====================================================================
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """
        ====================================================================
        세션 삭제
        ====================================================================

        Args:
            session_id: 삭제할 세션 ID

        Returns:
            bool: 삭제 성공 여부
                True: 세션이 존재했고 삭제됨
                False: 세션이 존재하지 않음
        ====================================================================
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """
        ====================================================================
        모든 세션 ID 목록
        ====================================================================

        Returns:
            List[str]: 현재 활성화된 세션 ID 리스트
        ====================================================================
        """
        return list(self._sessions.keys())

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """
        ====================================================================
        세션 요약
        ====================================================================

        Args:
            session_id: 조회할 세션 ID

        Returns:
            Optional[str]: 세션 요약 또는 None (세션 없음)
        ====================================================================
        """
        if session_id in self._sessions:
            return self._sessions[session_id].get_summary()
        return None


# ============================================================================
# 전역 대화 저장소
# ============================================================================
# 모듈 레벨에서 인스턴스를 생성하여 전역으로 사용합니다.
#
# 모듈 레벨 인스턴스의 특징:
#   - 모듈이 처음 import될 때 한 번만 생성
#   - 모든 import에서 같은 인스턴스 공유
#   - 프로세스 수명 동안 유지
#
# 사용:
#   from assistant.memory.conversation import conversation_store
#   memory = conversation_store.get_session("user-123")
#
# 주의:
#   - 멀티프로세스에서는 각 프로세스마다 별도 인스턴스
#   - 프로덕션에서는 Redis 등 공유 저장소 사용 권장
# ============================================================================
conversation_store = ConversationStore()
