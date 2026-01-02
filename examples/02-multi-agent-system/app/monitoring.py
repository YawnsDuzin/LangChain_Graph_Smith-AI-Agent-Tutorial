# app/monitoring.py
from langsmith import Client
from datetime import datetime, timedelta
from typing import Dict, List


class MultiAgentMonitor:
    """멀티 에이전트 시스템 모니터링"""

    def __init__(self, project_name: str = "multi-agent-tutorial"):
        self.client = Client()
        self.project_name = project_name

    def get_agent_performance(self, hours: int = 24) -> Dict:
        """에이전트별 성능 분석"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours)
        ))

        agent_stats = {
            "researcher": {"count": 0, "avg_latency": 0, "errors": 0},
            "writer": {"count": 0, "avg_latency": 0, "errors": 0},
            "reviewer": {"count": 0, "avg_latency": 0, "errors": 0},
            "supervisor": {"count": 0, "avg_latency": 0, "errors": 0}
        }

        for run in runs:
            name = run.name.lower()
            for agent in agent_stats:
                if agent in name:
                    agent_stats[agent]["count"] += 1
                    if run.end_time and run.start_time:
                        latency = (run.end_time - run.start_time).total_seconds()
                        current_avg = agent_stats[agent]["avg_latency"]
                        count = agent_stats[agent]["count"]
                        agent_stats[agent]["avg_latency"] = (
                            (current_avg * (count - 1) + latency) / count
                        )
                    if run.error:
                        agent_stats[agent]["errors"] += 1
                    break

        return agent_stats

    def get_workflow_metrics(self, hours: int = 24) -> Dict:
        """워크플로우 메트릭"""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=datetime.now() - timedelta(hours=hours),
            is_root=True
        ))

        if not runs:
            return {"message": "데이터 없음"}

        total = len(runs)
        successful = sum(1 for r in runs if not r.error)
        total_latency = sum(
            (r.end_time - r.start_time).total_seconds()
            for r in runs
            if r.end_time and r.start_time
        )

        return {
            "total_workflows": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_workflow_time": total_latency / total if total > 0 else 0
        }

    def print_dashboard(self, hours: int = 24):
        """대시보드 출력"""
        agent_perf = self.get_agent_performance(hours)
        workflow_metrics = self.get_workflow_metrics(hours)

        print("\n" + "="*60)
        print(f"📊 멀티 에이전트 시스템 대시보드 (최근 {hours}시간)")
        print("="*60)

        print("\n📈 워크플로우 메트릭:")
        print(f"  총 실행 수: {workflow_metrics.get('total_workflows', 0)}")
        print(f"  성공률: {workflow_metrics.get('success_rate', 0):.1%}")
        print(f"  평균 실행 시간: {workflow_metrics.get('avg_workflow_time', 0):.1f}초")

        print("\n👥 에이전트별 성능:")
        for agent, stats in agent_perf.items():
            print(f"\n  {agent.capitalize()}:")
            print(f"    - 호출 수: {stats['count']}")
            print(f"    - 평균 지연: {stats['avg_latency']:.2f}초")
            print(f"    - 에러 수: {stats['errors']}")

        print("\n" + "="*60)


# 사용 예시
if __name__ == "__main__":
    monitor = MultiAgentMonitor()
    monitor.print_dashboard(hours=24)
