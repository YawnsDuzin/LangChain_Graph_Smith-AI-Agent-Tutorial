# tests/test_workflow.py
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import MultiAgentWorkflow
from graph.state import create_initial_state


class TestMultiAgentWorkflow:
    """멀티 에이전트 워크플로우 테스트"""

    @pytest.fixture
    def workflow(self):
        w = MultiAgentWorkflow()
        w.compile(with_memory=False)
        return w

    def test_workflow_initialization(self, workflow):
        """워크플로우 초기화 테스트"""
        assert workflow.supervisor is not None
        assert workflow.researcher is not None
        assert workflow.writer is not None
        assert workflow.reviewer is not None
        assert workflow.app is not None

    def test_initial_state_creation(self):
        """초기 상태 생성 테스트"""
        state = create_initial_state("테스트 작업")

        assert state["task"] == "테스트 작업"
        assert state["next_agent"] == "supervisor"
        assert state["iteration_count"] == 0

    def test_simple_workflow_run(self, workflow):
        """간단한 워크플로우 실행 테스트"""
        result = workflow.run(
            "AI의 간단한 설명을 작성해주세요.",
            thread_id="test-1"
        )

        assert "status" in result
        assert "iterations" in result
        assert result["iterations"] > 0

    def test_workflow_streaming(self, workflow):
        """스트리밍 실행 테스트"""
        events = list(workflow.stream(
            "테스트 콘텐츠 작성",
            thread_id="test-2"
        ))

        assert len(events) > 0


class TestAgents:
    """개별 에이전트 테스트"""

    def test_researcher_agent(self):
        """연구 에이전트 테스트"""
        from agents.researcher import ResearcherAgent

        researcher = ResearcherAgent()
        assert researcher.name == "Researcher"

        state = {
            "task": "Python 프로그래밍에 대해 조사",
            "messages": []
        }

        result = researcher.invoke(state)
        assert "research_data" in result
        assert "messages" in result

    def test_writer_agent(self):
        """작성 에이전트 테스트"""
        from agents.writer import WriterAgent

        writer = WriterAgent()
        assert writer.name == "Writer"

        state = {
            "task": "Python 소개 글 작성",
            "messages": [],
            "research_data": {
                "topic": "Python",
                "findings": "Python은 인기 있는 프로그래밍 언어입니다.",
                "sources": [],
                "confidence": 0.9
            }
        }

        result = writer.invoke(state)
        assert "draft_content" in result

    def test_reviewer_agent(self):
        """검토 에이전트 테스트"""
        from agents.reviewer import ReviewerAgent

        reviewer = ReviewerAgent()
        assert reviewer.name == "Reviewer"

        state = {
            "task": "글 검토",
            "messages": [],
            "draft_content": {
                "title": "Python 소개",
                "sections": [{"title": "소개", "content": "..."}],
                "word_count": 500
            }
        }

        result = reviewer.invoke(state)
        assert "review_feedback" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
