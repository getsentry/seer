from unittest.mock import MagicMock, patch

import pytest

from seer.automation.agent.models import (
    LlmGenerateStructuredResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Usage,
)
from seer.automation.summarize.replays import (
    CommonReplaySummary,
    Replay,
    ReplayEvent,
    ReplaySummary,
    Step,
    SummarizeReplaysRequest,
    SummarizeReplaysResponse,
    run_cross_session_completion,
    run_single_replay_summary,
    summarize_replays,
)


class TestSummarizeReplays:
    @pytest.fixture
    def mock_llm_client(self):
        return MagicMock()

    @pytest.fixture
    def sample_replay(self):
        return Replay(
            events=[
                ReplayEvent(message="User clicked button", category="ui", type="click"),
                ReplayEvent(
                    message="Error occurred",
                    category="error",
                    type="error",
                    data={"error_message": "Test error"},
                ),
            ]
        )

    @pytest.fixture
    def sample_request(self, sample_replay):
        return SummarizeReplaysRequest(group_id=1, replays=[sample_replay])

    @patch("seer.automation.summarize.replays.run_single_replay_summary")
    @patch("seer.automation.summarize.replays.run_cross_session_completion")
    def test_summarize_replays(self, mock_cross_session, mock_single_summary, sample_request):
        mock_single_summary.return_value = ReplaySummary(
            user_steps_taken=[
                Step(description="User action", referenced_ids=[1], error_group_ids=[])
            ],
            pages_visited=["Home"],
            user_journey_summary="User journey",
        )
        mock_cross_session.return_value = MagicMock(
            common_user_steps_taken=[],
            reproduction_steps="Reproduction steps",
            issue_impact_summary="Impact summary",
        )

        result = summarize_replays(sample_request)

        assert isinstance(result, SummarizeReplaysResponse)
        assert result.reproduction == "Reproduction steps"
        assert result.impact_summary == "Impact summary"

        mock_single_summary.assert_called_once()
        mock_cross_session.assert_called_once()

    def test_run_single_replay_summary(self, mock_llm_client, sample_replay):
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=ReplaySummary(
                user_steps_taken=[], pages_visited=[], user_journey_summary="Test summary"
            ),
            metadata=LlmResponseMetadata(
                model="test",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = run_single_replay_summary(sample_replay, llm_client=mock_llm_client)

        assert isinstance(result, ReplaySummary)
        assert result.user_journey_summary == "Test summary"
        mock_llm_client.generate_structured.assert_called_once()

    def test_run_cross_session_completion(self, mock_llm_client):

        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=CommonReplaySummary(
                common_user_steps_taken=[],
                reproduction_steps="Test steps",
                issue_impact_summary="Test impact",
            ),
            metadata=LlmResponseMetadata(
                model="test",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = run_cross_session_completion(["Step 1", "Step 2"], llm_client=mock_llm_client)

        assert isinstance(result, CommonReplaySummary)
        assert result.reproduction_steps == "Test steps"
        assert result.issue_impact_summary == "Test impact"
        mock_llm_client.generate_structured.assert_called_once()
