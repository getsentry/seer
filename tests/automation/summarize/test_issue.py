from unittest.mock import MagicMock, Mock, patch

import pytest
from johen import generate

from seer.automation.models import IssueDetails
from seer.automation.summarize.issue import IssueSummary, run_summarize_issue, summarize_issue
from seer.automation.summarize.models import SummarizeIssueRequest, SummarizeIssueResponse


class TestSummarizeIssue:
    @pytest.fixture
    def mock_gpt_client(self):
        return Mock()

    @pytest.fixture
    def sample_request(self):
        iterator = generate(IssueDetails)
        return SummarizeIssueRequest(
            group_id=1, issue=next(iterator), connected_issues=[next(iterator), next(iterator)]
        )

    def test_summarize_issue_success(self, mock_gpt_client, sample_request):
        mock_structured_completion = MagicMock()
        mock_raw_summary = MagicMock(
            reason_step_by_step=[],
            summary_of_the_issue_based_on_your_step_by_step_reasoning="Test summary",
            summary_of_the_functionality_affected="Test functionality",
            five_to_ten_word_headline="Test headline",
        )
        mock_structured_completion.choices[0].message.parsed = mock_raw_summary
        mock_structured_completion.choices[0].message.refusal = None
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = (
            mock_structured_completion
        )

        result, raw_result = summarize_issue(sample_request, gpt_client=mock_gpt_client)

        assert isinstance(result, SummarizeIssueResponse)
        assert result.group_id == 1
        assert result.summary == "Test summary"
        assert result.impact == "Test functionality"
        assert result.headline == "Test headline"
        assert raw_result == mock_raw_summary

    def test_summarize_issue_refusal(self, mock_gpt_client, sample_request):
        mock_structured_completion = MagicMock()
        mock_structured_completion.choices[0].message.refusal = "I refuse!"
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = (
            mock_structured_completion
        )

        with pytest.raises(RuntimeError, match="I refuse!"):
            summarize_issue(sample_request, gpt_client=mock_gpt_client)

    def test_summarize_issue_parsing_failure(self, mock_gpt_client, sample_request):

        mock_structured_completion = MagicMock()
        mock_structured_completion.choices[0].message.parsed = None
        mock_structured_completion.choices[0].message.refusal = None
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = (
            mock_structured_completion
        )

        with pytest.raises(RuntimeError, match="Failed to parse message"):
            summarize_issue(sample_request, gpt_client=mock_gpt_client)

    @patch("seer.automation.summarize.issue.EventDetails.from_event")
    def test_summarize_issue_event_details(self, mock_from_event, mock_gpt_client, sample_request):
        mock_event_details = Mock()
        mock_event_details.format_event.side_effect = ["foo details", "bar details", "baz details"]
        mock_from_event.return_value = mock_event_details

        mock_structured_completion = MagicMock()
        mock_structured_completion.choices[0].message.parsed = MagicMock(
            reason_step_by_step=[],
            summary_of_the_issue_based_on_your_step_by_step_reasoning="Test summary",
            summary_of_the_functionality_affected="Test functionality",
            five_to_ten_word_headline="Test headline",
        )
        mock_structured_completion.choices[0].message.refusal = None
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = (
            mock_structured_completion
        )

        summarize_issue(sample_request, gpt_client=mock_gpt_client)

        mock_from_event.assert_any_call(sample_request.issue.events[0])
        mock_from_event.assert_any_call(sample_request.connected_issues[0].events[0])
        mock_from_event.assert_any_call(sample_request.connected_issues[1].events[0])
        assert mock_event_details.format_event.call_count == 3
        assert (
            "foo details"
            in mock_gpt_client.openai_client.beta.chat.completions.parse.call_args[1]["messages"][
                0
            ]["content"]
        )
        assert (
            "bar details"
            in mock_gpt_client.openai_client.beta.chat.completions.parse.call_args[1]["messages"][
                0
            ]["content"]
        )
        assert (
            "baz details"
            in mock_gpt_client.openai_client.beta.chat.completions.parse.call_args[1]["messages"][
                0
            ]["content"]
        )


class TestRunSummarizeIssue:
    @patch("seer.automation.summarize.issue.summarize_issue")
    def test_run_summarize_issue_langfuse_metadata(self, mock_summarize_issue):
        mock_summarize_issue.return_value = SummarizeIssueResponse(
            group_id=1, headline="headline", summary="summary", impact="impact"
        ), IssueSummary(
            reason_step_by_step=[],
            summary_of_the_issue_based_on_your_step_by_step_reasoning="summary",
            summary_of_the_functionality_affected="impact",
            five_to_ten_word_headline="headline",
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
        mock_summarize_issue.return_value = SummarizeIssueResponse(
            group_id=1, headline="headline", summary="summary", impact="impact"
        ), IssueSummary(
            reason_step_by_step=[],
            summary_of_the_issue_based_on_your_step_by_step_reasoning="summary",
            summary_of_the_functionality_affected="impact",
            five_to_ten_word_headline="headline",
        )

        # Create a sample request without organization_slug
        request = SummarizeIssueRequest(
            group_id=123, issue=next(generate(IssueDetails)), organization_id=456, project_id=789
        )

        # Call the function
        run_summarize_issue(request)

        # Assert that summarize_issue was called with the correct langfuse metadata
        mock_summarize_issue.assert_called_once()
        call_kwargs = mock_summarize_issue.call_args[1]

        assert call_kwargs["langfuse_tags"] == ["project:789", "group:123"]
        assert call_kwargs["langfuse_session_id"] == "group:123"
        assert call_kwargs["langfuse_user_id"] is None
