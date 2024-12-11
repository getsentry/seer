from unittest.mock import MagicMock, Mock, patch

import pytest
from johen import generate

from seer.automation.agent.client import LlmGenerateStructuredResponse, LlmResponseMetadata
from seer.automation.agent.models import LlmProviderType, Usage
from seer.automation.models import IssueDetails
from seer.automation.summarize.issue import IssueSummary, run_summarize_issue, summarize_issue
from seer.automation.summarize.models import SummarizeIssueRequest, SummarizeIssueResponse


class TestSummarizeIssue:
    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def sample_request(self):
        iterator = generate(IssueDetails)
        return SummarizeIssueRequest(
            group_id=1, issue=next(iterator), connected_issues=[next(iterator), next(iterator)]
        )

    def test_summarize_issue_success(self, mock_llm_client, sample_request):
        mock_structured_completion = MagicMock()
        mock_raw_summary = IssueSummary(
            title="Test headline",
            whats_wrong="Test what's wrong",
            session_related_issues="Test session related issues",
            possible_cause="Test possible cause",
        )
        mock_structured_completion.choices[0].message.parsed = mock_raw_summary
        mock_structured_completion.choices[0].message.refusal = None
        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=mock_raw_summary,
            metadata=LlmResponseMetadata(
                model="gpt-4o-mini-2024-07-18",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ),
        )

        result, raw_result = summarize_issue(sample_request, llm_client=mock_llm_client)

        assert isinstance(result, SummarizeIssueResponse)
        assert result.group_id == 1
        assert result.headline == "Test headline"
        assert result.whats_wrong == "Test what's wrong"
        assert result.trace == "Test session related issues"
        assert result.possible_cause == "Test possible cause"
        assert raw_result == mock_raw_summary

    @patch("seer.automation.summarize.issue.EventDetails.from_event")
    def test_summarize_issue_event_details(self, mock_from_event, mock_llm_client, sample_request):
        mock_event_details = Mock()
        mock_event_details.format_event.side_effect = ["foo details", "bar details", "baz details"]
        mock_from_event.return_value = mock_event_details

        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=IssueSummary(
                title="Test headline",
                whats_wrong="Test what's wrong",
                session_related_issues="Test session related issues",
                possible_cause="Test possible cause",
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.OPENAI,
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
