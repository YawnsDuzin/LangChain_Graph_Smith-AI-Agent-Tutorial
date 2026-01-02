# app/main.py
import os
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from graph.workflow import MultiAgentWorkflow

load_dotenv()


def print_header():
    print("\n" + "="*60)
    print("🤖 멀티 에이전트 콘텐츠 제작 시스템")
    print("="*60)
    print("\n에이전트 팀:")
    print("  📚 Researcher - 정보 조사 및 연구")
    print("  ✍️  Writer    - 콘텐츠 작성")
    print("  ✅ Reviewer   - 콘텐츠 검토")
    print("  👔 Supervisor - 워크플로우 조율")
    print("-"*60)


def run_interactive():
    """대화형 모드 실행"""
    print_header()

    workflow = MultiAgentWorkflow()
    workflow.compile()

    print("\n작업을 입력하세요. 종료하려면 'quit'를 입력하세요.\n")

    while True:
        try:
            task = input("📝 작업: ").strip()

            if not task:
                continue

            if task.lower() in ['quit', 'exit', '종료']:
                print("\n👋 시스템을 종료합니다.")
                break

            print("\n🔄 멀티 에이전트가 작업을 시작합니다...\n")

            # 스트리밍 실행
            for event in workflow.stream(task):
                for node, update in event.items():
                    if "messages" in update:
                        for msg in update["messages"]:
                            name = getattr(msg, 'name', 'System')
                            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                            print(f"[{name}] {content}")
                    print()

            print("\n" + "="*60)
            print("✅ 작업이 완료되었습니다.")
            print("="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


def run_single_task(task: str):
    """단일 작업 실행"""
    print_header()

    workflow = MultiAgentWorkflow()
    result = workflow.run(task)

    print("\n" + "="*60)
    print("📊 실행 결과")
    print("="*60)

    print(f"\n상태: {result['status']}")
    print(f"총 반복 횟수: {result['iterations']}")

    if result['research_data']:
        print("\n📚 연구 결과:")
        print(f"  - 주제: {result['research_data'].get('topic', 'N/A')}")
        print(f"  - 신뢰도: {result['research_data'].get('confidence', 0):.0%}")

    if result['draft_content']:
        print("\n✍️  작성 결과:")
        print(f"  - 제목: {result['draft_content'].get('title', 'N/A')}")
        print(f"  - 단어 수: {result['draft_content'].get('word_count', 0)}")
        print(f"  - 섹션 수: {len(result['draft_content'].get('sections', []))}")

    if result['review_feedback']:
        print("\n✅ 검토 결과:")
        print(f"  - 점수: {result['review_feedback'].get('score', 0)}/100")
        print(f"  - 승인: {'예' if result['review_feedback'].get('approved') else '아니오'}")

        strengths = result['review_feedback'].get('strengths', [])
        if strengths:
            print("  - 강점:")
            for s in strengths[:3]:
                print(f"    • {s}")

    print("\n" + "="*60)


def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        # 커맨드라인 인자로 작업 전달
        task = " ".join(sys.argv[1:])
        run_single_task(task)
    else:
        # 대화형 모드
        run_interactive()


if __name__ == "__main__":
    main()
