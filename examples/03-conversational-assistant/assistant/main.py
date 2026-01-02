# assistant/main.py
import os
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from assistant.graph.assistant import AIAssistant
from assistant.memory.conversation import conversation_store


def print_header():
    print("\n" + "="*60)
    print("🤖 대화형 AI 어시스턴트")
    print("="*60)
    print("\n사용 가능한 기능:")
    print("  🔍 웹 검색: '~에 대해 검색해줘'")
    print("  🧮 계산기: '123 * 456 계산해줘'")
    print("  🐍 코드 실행: 'Python으로 ~해줘'")
    print("  📅 날짜/시간: '오늘 날짜가 뭐야?'")
    print("  📝 유틸리티: '텍스트 요약해줘'")
    print("\n명령어:")
    print("  /clear - 대화 기록 삭제")
    print("  /history - 대화 기록 보기")
    print("  /help - 도움말")
    print("  quit 또는 exit - 종료")
    print("-"*60 + "\n")


def print_help():
    print("""
📚 AI 어시스턴트 도움말

[검색 기능]
• "최신 AI 뉴스 알려줘" - 웹 검색
• "파이썬이 뭐야?" - 위키피디아 검색

[계산 기능]
• "123 + 456 계산해줘" - 기본 계산
• "100달러를 원으로" - 단위 변환
• "20%의 500은?" - 퍼센트 계산

[코드 기능]
• "피보나치 함수 만들어줘" - 코드 생성
• "이 코드 설명해줘" - 코드 설명
• "이 코드 실행해줘" - 코드 실행 (승인 필요)

[날짜/시간]
• "지금 몇 시야?" - 현재 시간
• "오늘부터 100일 후는?" - 날짜 계산

[유틸리티]
• "이 텍스트 요약해줘" - 텍스트 요약
""")


def run_cli():
    """CLI 모드 실행"""
    print_header()

    assistant = AIAssistant()
    assistant.compile()

    session_id = "cli-session"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # 명령어 처리
            if user_input.startswith("/"):
                cmd = user_input[1:].lower()

                if cmd == "clear":
                    assistant.clear_conversation(session_id)
                    print("✅ 대화 기록이 삭제되었습니다.\n")
                    continue

                elif cmd == "history":
                    history = assistant.get_conversation_history(session_id)
                    if not history:
                        print("대화 기록이 없습니다.\n")
                    else:
                        print("\n📜 대화 기록:")
                        for msg in history:
                            role = "You" if msg["role"] == "user" else "AI"
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"  {role}: {content}")
                        print()
                    continue

                elif cmd == "help":
                    print_help()
                    continue

                else:
                    print(f"알 수 없는 명령어: /{cmd}\n")
                    continue

            # 종료
            if user_input.lower() in ["quit", "exit", "종료"]:
                print("\n👋 안녕히 가세요!")
                break

            # 응답 생성
            print("\n", end="")
            response = assistant.chat(user_input, session_id=session_id)
            print(f"AI: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")


def run_web():
    """웹 서버 실행"""
    print("\n🌐 웹 서버를 시작합니다...")
    print("브라우저에서 http://localhost:8000 을 열어주세요.\n")

    import uvicorn
    from web.app import app
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "web":
            run_web()
        elif mode == "cli":
            run_cli()
        else:
            print(f"알 수 없는 모드: {mode}")
            print("사용법: python main.py [cli|web]")
    else:
        # 기본: CLI 모드
        run_cli()


if __name__ == "__main__":
    main()
