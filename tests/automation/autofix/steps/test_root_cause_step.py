import unittest
from unittest.mock import MagicMock, patch

from johen import generate

from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseAnalysisOutput,
)
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixRequestOptions,
)
from seer.automation.autofix.steps.root_cause_step import RootCauseStep
from seer.automation.models import IssueDetails, RepoDefinition, SentryEventData
from seer.automation.summarize.issue import IssueSummary


class TestRootCauseStep(unittest.TestCase):
    @patch("seer.automation.autofix.steps.root_cause_step.RootCauseAnalysisComponent")
    def test_happy_path(self, mock_root_cause_component):
        mock_context = MagicMock()
        RootCauseStep._instantiate_context = MagicMock(return_value=mock_context)

        error_event = next(generate(SentryEventData))

        mock_context.event_manager = MagicMock()
        mock_context.has_missing_codebase_indexes.return_value = False
        mock_context.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[error_event]),
                issue_summary=IssueSummary(
                    title="title",
                    whats_wrong="whats wrong",
                    session_related_issues="trace",
                    possible_cause="possible cause",
                ),
            )
        )

        mock_root_cause_output = next(generate(RootCauseAnalysisOutput))
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output

        step = RootCauseStep({"run_id": 1, "step_id": 1})

        step.invoke()

        step.context.event_manager.send_root_cause_analysis_start.assert_called()
        step.context.process_event_paths.assert_called()
        step.context.event_manager.send_root_cause_analysis_result.assert_called()

    @patch("seer.automation.autofix.steps.root_cause_step.RootCauseAnalysisComponent")
    def test_github_copilot_pr_comment(self, mock_root_cause_component):
        mock_context = MagicMock()
        RootCauseStep._instantiate_context = MagicMock(return_value=mock_context)

        error_event = next(generate(SentryEventData))
        pr_url = "https://github.com/example/repo/pull/123"
        repo = RepoDefinition(name="repo", owner="example", provider="github", external_id="123")

        mock_context.event_manager = MagicMock()
        mock_context.has_missing_codebase_indexes.return_value = False
        mock_context.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[repo],
                issue=IssueDetails(id=0, title="", events=[error_event]),
                issue_summary=IssueSummary(
                    title="title",
                    whats_wrong="whats wrong",
                    session_related_issues="trace",
                    possible_cause="possible cause",
                ),
                options=AutofixRequestOptions(comment_on_pr_with_url=pr_url),
            )
        )

        mock_root_cause_output = RootCauseAnalysisOutput(
            causes=[RootCauseAnalysisItem(description="Test root cause", id=0, title="Test title")]
        )
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output

        step = RootCauseStep({"run_id": 1, "step_id": 1})

        step.invoke()

        # Assert that the PR comment method was called with the correct arguments
        mock_context.comment_root_cause_on_pr.assert_called_once_with(
            pr_url=pr_url,
            repo_definition=repo,
            root_cause="# Test title\n\n## Description\nTest root cause",
        )

        # Assert that other expected methods were called
        step.context.event_manager.send_root_cause_analysis_start.assert_called()
        step.context.process_event_paths.assert_called()
        step.context.event_manager.send_root_cause_analysis_result.assert_called()
