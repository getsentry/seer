from unittest.mock import MagicMock, Mock, patch

import pytest
from johen import generate

from seer.automation.models import IssueDetails
from seer.automation.summarize.issue import summarize_issue
from seer.automation.summarize.models import SummarizeIssueRequest, SummarizeIssueResponse


class TestSummarizeIssue:
    @pytest.fixture
    def mock_gpt_client(self):
        return Mock()

    @pytest.fixture
    def sample_request(self):
        return SummarizeIssueRequest(group_id=1, issue=next(generate(IssueDetails)))

    def test_summarize_issue_success(self, mock_gpt_client, sample_request):
        mock_structured_completion = MagicMock()
        mock_structured_completion.choices[0].message.parsed = MagicMock(
            reason_step_by_step=[],
            summary_of_issue="Test summary",
            affects_what_functionality="Test functionality",
            known_customer_impact="Test impact",
            customer_impact_is_known=True,
        )
        mock_structured_completion.choices[0].message.refusal = None
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = (
            mock_structured_completion
        )

        result = summarize_issue(sample_request, gpt_client=mock_gpt_client)

        assert isinstance(result, SummarizeIssueResponse)
        assert result.group_id == 1
        assert result.summary == "Test summary"
        assert result.impact == "Test functionality Test impact"

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
        mock_event_details.format_event.return_value = "Formatted event details"
        mock_from_event.return_value = mock_event_details

        mock_structured_completion = MagicMock()
        mock_structured_completion.choices[0].message.parsed = MagicMock(
            reason_step_by_step=[],
            summary_of_issue="Test summary",
            affects_what_functionality="Test functionality",
            known_customer_impact="Test impact",
            customer_impact_is_known=False,
        )
        mock_structured_completion.choices[0].message.refusal = None
        mock_gpt_client.openai_client.beta.chat.completions.parse.return_value = (
            mock_structured_completion
        )

        summarize_issue(sample_request, gpt_client=mock_gpt_client)

        mock_from_event.assert_called_once_with(sample_request.issue.events[0])
        mock_event_details.format_event.assert_called_once()
        assert (
            "Formatted event details"
            in mock_gpt_client.openai_client.beta.chat.completions.parse.call_args[1]["messages"][
                0
            ]["content"]
        )
