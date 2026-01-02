# app/monitoring.py
from langsmith import Client
from datetime import datetime, timedelta
from typing import Optional


class RAGMonitor:
    """RAG 시스템 모니터링"""

    def __init__(self, project_name: str = None):
        self.client = Client()
        self.project_name = project_name or "rag-chatbot-tutorial"

    def get_recent_runs(self, hours: int = 24, limit: int = 100):
        """최근 런 조회"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),
            is_root=True,
            limit=limit
        ))
        return runs

    def get_error_runs(self, hours: int = 24):
        """에러 런 조회"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),
            is_root=True,
            error=True
        ))
        return runs

    def calculate_metrics(self, hours: int = 24):
        """메트릭 계산"""
        runs = self.get_recent_runs(hours)

        if not runs:
            return None

        # 지연시간 계산
        latencies = []
        costs = []
        errors = 0

        for run in runs:
            if run.end_time and run.start_time:
                latency = (run.end_time - run.start_time).total_seconds()
                latencies.append(latency)

            if run.total_cost:
                costs.append(run.total_cost)

            if run.error:
                errors += 1

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
        """리포트 출력"""
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
        """피드백 추가"""
        self.client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=score,
            comment=comment
        )


# 사용 예시
if __name__ == "__main__":
    monitor = RAGMonitor()
    monitor.print_report(hours=24)
