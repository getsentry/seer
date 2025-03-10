from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from johen import generate

from seer.automation.agent.client import LlmGenerateStructuredResponse, LlmResponseMetadata
from seer.automation.agent.models import LlmProviderType, Usage
from seer.automation.models import IssueDetails
from seer.automation.summarize.issue import (
    IssueSummary,
    IssueSummaryForLlmToGenerate,
    IssueSummaryWithScores,
    run_summarize_issue,
    summarize_issue,
)
from seer.automation.summarize.models import (
    SummarizeIssueRequest,
    SummarizeIssueResponse,
    SummarizeIssueScores,
)


class TestSummarizeIssue:
    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def sample_request(self):
        issues_dir = Path(__file__).parent / "fixtures" / "issues"
        issues: list[IssueDetails] = []
        for path in issues_dir.glob("issue_to_summarize*.json"):
            with path.open() as f:
                issues.append(IssueDetails.model_validate_json(f.read()))
        assert len(issues) >= 2, "Need at least 2 issues so that there's a connected issue"
        return SummarizeIssueRequest(
            group_id=123,
            issue=issues[0],
            connected_issues=issues[1:],
            organization_id=456,
            organization_slug="test-org",
            project_id=789,
        )

    @pytest.mark.vcr()
    def test_summarize_issue_success(self, sample_request, score_num_decimal_places: int = 10):
        result, raw_result = summarize_issue(sample_request)

        expected_raw_result = IssueSummaryWithScores(
            title="Multiple Issues in Session: Shared Resource Failure?",
            whats_wrong="**red-timothy-sandwich** issue occurred.",
            session_related_issues="Other issues: **cyan-vincent-banana** and **green-fred-tennis** occurred in the same session.",
            possible_cause="Perhaps a **shared resource** or **dependency** is failing, affecting multiple features.",
            scores=SummarizeIssueScores(possible_cause_confidence=0.5, possible_cause_novelty=0.8),
        )
        # Round for some tolerance during equality comparison.
        raw_result.scores.possible_cause_confidence = round(
            raw_result.scores.possible_cause_confidence, score_num_decimal_places
        )
        raw_result.scores.possible_cause_novelty = round(
            raw_result.scores.possible_cause_novelty, score_num_decimal_places
        )
        assert raw_result == expected_raw_result

        assert isinstance(result, SummarizeIssueResponse)
        assert result.group_id == 123
        assert result.headline == expected_raw_result.title
        assert result.whats_wrong == expected_raw_result.whats_wrong
        assert result.trace == expected_raw_result.session_related_issues
        assert result.possible_cause == expected_raw_result.possible_cause
        assert result.scores is None

    @patch("seer.automation.summarize.issue.EventDetails.from_event")
    def test_summarize_issue_event_details(self, mock_from_event, mock_llm_client, sample_request):
        mock_event_details = Mock()
        mock_event_details.format_event.side_effect = ["foo details", "bar details", "baz details"]
        mock_from_event.return_value = mock_event_details

        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=IssueSummaryForLlmToGenerate(
                whats_wrong="Test what's wrong",
                session_related_issues="Test session related issues",
                possible_cause="Test possible cause",
                possible_cause_novelty_score=0.5,
                possible_cause_confidence_score=0.5,
                title="Test headline",
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ),
        )

        summarize_issue(sample_request, llm_client=mock_llm_client)

        mock_from_event.assert_any_call(sample_request.issue.events[0])
        mock_from_event.assert_any_call(sample_request.connected_issues[0].events[0])
        mock_from_event.assert_any_call(sample_request.connected_issues[1].events[0])
        assert mock_event_details.format_event.call_count == 3
        assert "foo details" in mock_llm_client.generate_structured.call_args[1]["prompt"]
        assert "bar details" in mock_llm_client.generate_structured.call_args[1]["prompt"]
        assert "baz details" in mock_llm_client.generate_structured.call_args[1]["prompt"]


class TestRunSummarizeIssue:
    @patch("seer.automation.summarize.issue.summarize_issue")
    def test_run_summarize_issue_langfuse_metadata(self, mock_summarize_issue):
        mock_summarize_issue.return_value = (
            SummarizeIssueResponse(
                group_id=1,
                headline="Test headline",
                whats_wrong="Test what's wrong",
                trace="Test trace",
                possible_cause="Test possible cause",
                scores=SummarizeIssueScores(
                    possible_cause_confidence=0.5,
                    possible_cause_novelty=0.5,
                ),
            ),
            IssueSummary(
                title="Test headline",
                whats_wrong="Test what's wrong",
                session_related_issues="Test session related issues",
                possible_cause="Test possible cause",
            ),
        )

        # Create a sample request
        request = SummarizeIssueRequest(
            group_id=123,
            issue=next(generate(IssueDetails)),
            organization_id=456,
            organization_slug="test-org",
            project_id=789,
        )

        # Call the function
        run_summarize_issue(request)

        # Assert that summarize_issue was called with the correct langfuse metadata
        mock_summarize_issue.assert_called_once()
        call_kwargs = mock_summarize_issue.call_args[1]

        assert call_kwargs["langfuse_tags"] == ["org:test-org", "project:789", "group:123"]
        assert call_kwargs["langfuse_session_id"] == "group:123"
        assert call_kwargs["langfuse_user_id"] == "org:test-org"

    @patch("seer.automation.summarize.issue.summarize_issue")
    def test_run_summarize_issue_langfuse_metadata_no_org_slug(self, mock_summarize_issue):
        mock_summarize_issue.return_value = (
            SummarizeIssueResponse(
                group_id=1,
                headline="Test headline",
                whats_wrong="Test what's wrong",
                trace="Test trace",
                possible_cause="Test possible cause",
                scores=None,
            ),
            IssueSummary(
                title="Test headline",
                whats_wrong="Test what's wrong",
                session_related_issues="Test session related issues",
                possible_cause="Test possible cause",
            ),
        )

        # Create a sample request without organization_slug
        request = SummarizeIssueRequest(
            group_id=123,
            issue=next(generate(IssueDetails)),
            organization_id=456,
            project_id=789,
        )

        # Call the function
        run_summarize_issue(request)

        # Assert that summarize_issue was called with the correct langfuse metadata
        mock_summarize_issue.assert_called_once()
        call_kwargs = mock_summarize_issue.call_args[1]

        assert call_kwargs["langfuse_tags"] == ["project:789", "group:123"]
        assert call_kwargs["langfuse_session_id"] == "group:123"
        assert call_kwargs["langfuse_user_id"] is None
