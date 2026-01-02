# app/monitoring.py
"""
LangSmith 모니터링 모듈

이 모듈은 LangSmith를 사용하여 RAG 챗봇의 성능을 모니터링합니다.
실행 기록 조회, 메트릭 분석, 피드백 수집 기능을 제공합니다.

LangSmith 개념:
---------------
LangSmith는 LLM 애플리케이션의 개발, 모니터링, 평가를 위한 플랫폼입니다.

주요 기능:
1. 트레이싱 (Tracing):
   - 모든 LLM 호출 자동 기록
   - 입력, 출력, 지연시간, 비용 추적
   - 중첩된 호출 시각화

2. 평가 (Evaluation):
   - 데이터셋 기반 자동 테스트
   - 커스텀 평가 지표
   - 회귀 테스트

3. 모니터링 (Monitoring):
   - 실시간 대시보드
   - 에러 추적
   - 성능 분석

4. 프롬프트 관리 (Prompt Hub):
   - 프롬프트 버전 관리
   - 팀 공유
   - A/B 테스트

LangSmith 설정 방법:
-------------------
1. https://smith.langchain.com 에서 계정 생성
2. API 키 발급
3. 환경 변수 설정:
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-api-key
   LANGCHAIN_PROJECT=your-project-name

자동 트레이싱:
- 환경 변수 설정 시 LangChain 호출이 자동으로 기록됨
- 코드 수정 없이 모니터링 가능
- langsmith.traceable 데코레이터로 커스텀 함수 추적 가능
"""

# =============================================================================
# LangSmith 임포트 설명
# =============================================================================

# langsmith.Client
# -----------------
# LangSmith API 클라이언트입니다.
# 프로그래밍 방식으로 LangSmith 기능에 접근합니다.
#
# 주요 메서드:
# - list_runs(): 실행 기록 조회
# - create_dataset(): 데이터셋 생성
# - create_feedback(): 피드백 추가
# - run_on_dataset(): 평가 실행
#
# 인증:
# - 환경변수 LANGCHAIN_API_KEY 자동 사용
# - 또는 Client(api_key="...") 로 직접 지정
#
# 사용 예시:
#   from langsmith import Client
#
#   client = Client()
#   runs = client.list_runs(project_name="my-project")
from langsmith import Client

from datetime import datetime, timedelta
from typing import Optional


class RAGMonitor:
    """
    RAG 시스템 모니터링 클래스

    LangSmith API를 통해 RAG 챗봇의 실행 기록을 분석합니다.

    제공 기능:
    - 최근 실행 기록 조회
    - 에러 실행 필터링
    - 성능 메트릭 계산
    - 사용자 피드백 수집

    사용 예시:
        monitor = RAGMonitor(project_name="rag-chatbot")

        # 리포트 출력
        monitor.print_report(hours=24)

        # 메트릭 조회
        metrics = monitor.calculate_metrics(hours=24)
        print(f"평균 지연시간: {metrics['avg_latency']:.2f}초")

        # 피드백 추가
        monitor.add_feedback(run_id="...", score=1.0, comment="정확한 답변")
    """

    def __init__(self, project_name: str = None):
        """
        RAGMonitor 초기화

        Args:
            project_name (str, optional): LangSmith 프로젝트 이름
                - None이면 기본값 "rag-chatbot-tutorial" 사용
                - 환경변수 LANGCHAIN_PROJECT와 일치해야 함

        Client 초기화 상세 설명:
        ------------------------
        Client 생성자:
        - api_key: LangSmith API 키 (선택, 환경변수 우선)
        - api_url: API 엔드포인트 (기본: https://api.smith.langchain.com)

        프로젝트:
        - 관련 실행들을 그룹화하는 논리적 단위
        - 대시보드에서 프로젝트별 필터링 가능
        - 개발/스테이징/프로덕션 환경 분리에 유용
        """
        # LangSmith 클라이언트 생성
        # API 키는 환경변수 LANGCHAIN_API_KEY에서 자동 로드
        self.client = Client()

        # 프로젝트 이름 설정
        self.project_name = project_name or "rag-chatbot-tutorial"

    def get_recent_runs(self, hours: int = 24, limit: int = 100):
        """
        최근 런 조회

        지정된 시간 내의 실행 기록을 조회합니다.

        Args:
            hours (int): 조회할 시간 범위 (최근 N시간)
            limit (int): 최대 반환 건수

        Returns:
            list: Run 객체 리스트

        list_runs() 상세 설명:
        -----------------------
        LangSmith에서 실행 기록을 조회합니다.

        파라미터:
        - project_name: 프로젝트 이름으로 필터링
        - start_time: 시작 시간 이후의 런만 조회
        - is_root: True면 최상위 런만 (중첩된 하위 런 제외)
        - limit: 최대 결과 수
        - run_type: "llm", "chain", "tool" 등으로 필터링
        - error: True면 에러 런만 조회

        Run 객체 주요 속성:
        - id: 고유 식별자
        - name: 런 이름
        - run_type: 런 유형 (llm, chain, tool)
        - inputs: 입력 데이터
        - outputs: 출력 데이터
        - error: 에러 메시지 (있는 경우)
        - start_time: 시작 시간
        - end_time: 종료 시간
        - total_tokens: 사용된 총 토큰 수
        - total_cost: 비용 (USD)
        """
        # list_runs 호출
        # Generator를 list로 변환
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),  # 시작 시간
            is_root=True,    # 최상위 런만 (중첩 제외)
            limit=limit      # 최대 개수
        ))
        return runs

    def get_error_runs(self, hours: int = 24):
        """
        에러 런 조회

        에러가 발생한 실행만 조회합니다.

        Args:
            hours (int): 조회할 시간 범위

        Returns:
            list: 에러가 발생한 Run 객체 리스트

        에러 분석 활용:
        - 빈번한 에러 패턴 파악
        - 입력 데이터 분석
        - 재현 및 디버깅
        """
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),
            is_root=True,
            error=True  # 에러 런만 필터링
        ))
        return runs

    def calculate_metrics(self, hours: int = 24):
        """
        메트릭 계산

        최근 실행 기록을 분석하여 주요 지표를 계산합니다.

        Args:
            hours (int): 분석할 시간 범위

        Returns:
            dict: 계산된 메트릭
                - total_runs: 총 실행 수
                - avg_latency: 평균 응답 시간 (초)
                - max_latency: 최대 응답 시간 (초)
                - min_latency: 최소 응답 시간 (초)
                - total_cost: 총 비용 (USD)
                - error_count: 에러 수
                - error_rate: 에러율 (0~1)

        메트릭 해석:
        -----------
        avg_latency:
        - 1초 이하: 매우 빠름
        - 1~3초: 양호
        - 3~5초: 보통
        - 5초 이상: 개선 필요

        error_rate:
        - 0.01 이하: 양호
        - 0.01~0.05: 주의 필요
        - 0.05 이상: 즉시 조치 필요
        """
        runs = self.get_recent_runs(hours)

        if not runs:
            return None

        # 메트릭 수집
        latencies = []  # 지연시간
        costs = []      # 비용
        errors = 0      # 에러 수

        for run in runs:
            # 지연시간 계산 (종료-시작)
            if run.end_time and run.start_time:
                latency = (run.end_time - run.start_time).total_seconds()
                latencies.append(latency)

            # 비용 수집
            if run.total_cost:
                costs.append(run.total_cost)

            # 에러 카운트
            if run.error:
                errors += 1

        # 메트릭 반환
        return {
            "total_runs": len(runs),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "total_cost": sum(costs),
            "error_count": errors,
            "error_rate": errors / len(runs) if runs else 0
        }

    def print_report(self, hours: int = 24):
        """
        리포트 출력

        콘솔에 성능 리포트를 출력합니다.

        Args:
            hours (int): 분석할 시간 범위
        """
        metrics = self.calculate_metrics(hours)

        if not metrics:
            print("데이터가 없습니다.")
            return

        print("\n" + "="*50)
        print(f"📊 RAG 챗봇 모니터링 리포트 (최근 {hours}시간)")
        print("="*50)
        print(f"총 런 수: {metrics['total_runs']}")
        print(f"평균 지연시간: {metrics['avg_latency']:.2f}초")
        print(f"최대 지연시간: {metrics['max_latency']:.2f}초")
        print(f"최소 지연시간: {metrics['min_latency']:.2f}초")
        print(f"총 비용: ${metrics['total_cost']:.4f}")
        print(f"에러 수: {metrics['error_count']}")
        print(f"에러율: {metrics['error_rate']:.1%}")
        print("="*50 + "\n")

    def add_feedback(self, run_id: str, score: float, comment: str = None):
        """
        피드백 추가

        특정 실행에 사용자 피드백을 추가합니다.

        Args:
            run_id (str): 피드백을 추가할 런의 ID
            score (float): 점수 (0.0 ~ 1.0)
            comment (str, optional): 코멘트

        create_feedback() 상세 설명:
        -----------------------------
        LangSmith에 피드백을 기록합니다.

        피드백 유형:
        - 사용자 평가: 최종 사용자가 제출한 만족도
        - 자동 평가: 프로그래밍 방식으로 생성된 점수
        - 전문가 리뷰: 내부 QA 팀의 평가

        score 해석:
        - 1.0: 완벽, 정확한 응답
        - 0.7~0.9: 좋음, 약간의 개선 여지
        - 0.4~0.6: 보통, 개선 필요
        - 0.0~0.3: 나쁨, 잘못된 응답

        피드백 활용:
        - 평가 데이터셋 생성
        - 모델 Fine-tuning
        - 품질 추세 분석
        - A/B 테스트 비교

        사용 예시:
            # 긍정적 피드백
            monitor.add_feedback(
                run_id="run-123",
                score=1.0,
                comment="정확하고 도움이 되는 답변"
            )

            # 부정적 피드백
            monitor.add_feedback(
                run_id="run-456",
                score=0.2,
                comment="질문과 관련 없는 답변"
            )
        """
        # create_feedback 호출
        self.client.create_feedback(
            run_id=run_id,       # 런 ID
            key="user_rating",   # 피드백 키 (유형)
            score=score,         # 점수
            comment=comment      # 코멘트
        )


# =============================================================================
# 추가 LangSmith 기능 예시
# =============================================================================

def example_traceable_function():
    """
    traceable 데코레이터 사용 예시

    @traceable 데코레이터를 사용하면 일반 함수도 LangSmith에서 추적됩니다.

    사용법:
        from langsmith import traceable

        @traceable
        def my_function(input_text: str) -> str:
            # 처리 로직
            return result

    traceable 파라미터:
    - name: 런 이름 (기본: 함수명)
    - run_type: 런 유형 ("chain", "tool", "llm", "prompt")
    - metadata: 추가 메타데이터

    고급 사용:
        @traceable(name="custom_name", run_type="tool")
        def my_tool(query: str) -> dict:
            return {"result": query.upper()}
    """
    pass


def example_dataset_evaluation():
    """
    데이터셋 평가 예시

    LangSmith에서 데이터셋을 생성하고 평가를 실행하는 방법입니다.

    1. 데이터셋 생성:
        from langsmith import Client

        client = Client()
        dataset = client.create_dataset(
            dataset_name="rag-eval-dataset",
            description="RAG 챗봇 평가용 데이터셋"
        )

    2. 예시 추가:
        client.create_example(
            inputs={"question": "LangChain이란?"},
            outputs={"answer": "LLM 프레임워크입니다."},
            dataset_id=dataset.id
        )

    3. 평가 실행:
        from langsmith.evaluation import evaluate

        def predict(inputs):
            # RAG 체인 실행
            return {"answer": rag_chain.invoke(inputs["question"])}

        def accuracy_evaluator(run, example):
            # 정확도 평가
            predicted = run.outputs["answer"]
            expected = example.outputs["answer"]
            score = calculate_similarity(predicted, expected)
            return {"score": score}

        results = evaluate(
            predict,
            data=dataset_name,
            evaluators=[accuracy_evaluator]
        )
    """
    pass


# =============================================================================
# 사용 예시 및 테스트
# =============================================================================
if __name__ == "__main__":
    """
    모니터링 테스트

    실행 방법:
        python -m app.monitoring

    참고: LANGCHAIN_API_KEY 환경변수 필요
    """
    try:
        monitor = RAGMonitor()

        # 리포트 출력
        monitor.print_report(hours=24)

        # 최근 런 조회
        print("=== 최근 5개 런 ===")
        runs = monitor.get_recent_runs(hours=24, limit=5)
        for run in runs:
            print(f"- {run.name}: {run.run_type}")
            if run.total_cost:
                print(f"  비용: ${run.total_cost:.4f}")
            if run.error:
                print(f"  에러: {run.error[:50]}...")

    except Exception as e:
        print(f"LangSmith 연결 오류: {e}")
        print("LANGCHAIN_API_KEY 환경변수를 확인하세요.")
