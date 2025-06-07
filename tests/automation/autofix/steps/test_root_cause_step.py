import unittest
from unittest.mock import MagicMock, patch

from johen import generate

from seer.automation.agent.models import Message
from seer.automation.autofix.components.confidence import ConfidenceOutput
from seer.automation.autofix.components.root_cause.models import (
    RelevantCodeFile,
    RootCauseAnalysisItem,
    RootCauseAnalysisOutput,
    TimelineEvent,
)
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixRequestOptions,
    DefaultStep,
)
from seer.automation.autofix.models import RootCauseStep as RootCauseStepModel
from seer.automation.autofix.models import StepType
from seer.automation.autofix.steps.root_cause_step import RootCauseStep
from seer.automation.models import IssueDetails, RepoDefinition, SentryEventData
from seer.automation.summarize.issue import IssueSummary


class TestRootCauseStep(unittest.TestCase):
    def setUp(self):
        self.mock_tools = MagicMock()
        self.mock_solution_instance = MagicMock()

        self.mock_tools.return_value.__enter__ = MagicMock(return_value=self.mock_tools)
        self.mock_tools.return_value.__exit__ = MagicMock(return_value=None)

        self.mock_solution_instance.return_value = self.mock_solution_instance

    def _create_base_mock_context(self):
        """Helper to create basic mock context with common setup"""
        mock_context = MagicMock()
        mock_context.event_manager = MagicMock()
        mock_context.has_missing_codebase_indexes.return_value = False
        RootCauseStep._instantiate_context = MagicMock(return_value=mock_context)
        return mock_context

    @patch("seer.automation.autofix.steps.root_cause_step.AutofixSolutionStep")
    @patch("seer.automation.autofix.steps.root_cause_step.BaseTools")
    @patch("seer.automation.autofix.steps.root_cause_step.RootCauseAnalysisComponent")
    def test_happy_path(self, mock_root_cause_component, mock_base_tools, mock_solution_step):
        mock_context = self._create_base_mock_context()

        error_event = next(generate(SentryEventData))

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
                options=AutofixRequestOptions(disable_interactivity=True),
            )
        )

        mock_root_cause_output = next(generate(RootCauseAnalysisOutput))
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output

        step = RootCauseStep({"run_id": 1, "step_id": 1})

        step.invoke()

        step.context.event_manager.send_root_cause_analysis_start.assert_called()
        step.context.process_event_paths.assert_called()
        step.context.event_manager.send_root_cause_analysis_result.assert_called()

    @patch("seer.automation.autofix.steps.root_cause_step.AutofixSolutionStep")
    @patch("seer.automation.autofix.steps.root_cause_step.BaseTools")
    @patch("seer.automation.autofix.steps.root_cause_step.RootCauseAnalysisComponent")
    def test_github_copilot_pr_comment(
        self, mock_root_cause_component, mock_base_tools, mock_solution_step
    ):
        mock_context = self._create_base_mock_context()

        error_event = next(generate(SentryEventData))
        pr_url = "https://github.com/example/repo/pull/123"
        repo = RepoDefinition(name="repo", owner="example", provider="github", external_id="123")

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
                options=AutofixRequestOptions(
                    comment_on_pr_with_url=pr_url, disable_interactivity=True
                ),
            )
        )

        mock_root_cause_output = RootCauseAnalysisOutput(
            causes=[
                RootCauseAnalysisItem(
                    id=0,
                    root_cause_reproduction=[
                        TimelineEvent(
                            title="Test title",
                            code_snippet_and_analysis="Test root cause",
                            timeline_item_type="internal_code",
                            relevant_code_file=RelevantCodeFile(
                                file_path="test.py", repo_name="owner/repo"
                            ),
                            is_most_important_event=True,
                        )
                    ],
                )
            ]
        )
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output
        step = RootCauseStep({"run_id": 1, "step_id": 1})

        step.invoke()

        # Assert that the PR comment method was called with the correct arguments
        mock_context.comment_root_cause_on_pr.assert_called_once_with(
            pr_url=pr_url,
            repo_definition=repo,
            root_cause="# Root Cause\n\n### Test title\nTest root cause",
        )

        # Assert that other expected methods were called
        step.context.event_manager.send_root_cause_analysis_start.assert_called()
        step.context.process_event_paths.assert_called()
        step.context.event_manager.send_root_cause_analysis_result.assert_called()

    @patch("seer.automation.autofix.steps.root_cause_step.AutofixSolutionStep")
    @patch("seer.automation.autofix.steps.root_cause_step.BaseTools")
    @patch("seer.automation.autofix.steps.root_cause_step.RootCauseAnalysisComponent")
    @patch("seer.automation.autofix.steps.root_cause_step.ConfidenceComponent")
    def test_confidence_evaluation(
        self,
        mock_confidence_component,
        mock_root_cause_component,
        mock_base_tools,
        mock_solution_step,
    ):
        mock_context = self._create_base_mock_context()

        # Create a test event
        error_event = next(generate(SentryEventData))

        # Create a proper DefaultStep object
        default_step = DefaultStep(key="default_step", title="Default Step")

        # Create a proper RootCauseStep object
        root_cause_step = RootCauseStepModel(
            key="root_cause_analysis",
            title="Root Cause Analysis",
            type=StepType.ROOT_CAUSE_ANALYSIS,
            causes=[],
        )

        # Set up the state with interactivity enabled
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
                options=AutofixRequestOptions(disable_interactivity=False),  # Enable interactivity
            ),
            steps=[default_step, root_cause_step],  # Add both steps to the steps list
        )

        # Mock the root cause output
        mock_root_cause_output = next(generate(RootCauseAnalysisOutput))
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output

        # Mock the confidence component output
        mock_confidence_output = ConfidenceOutput(
            question="This is a test question",
            output_confidence_score=0.85,
            proceed_confidence_score=0.75,
        )
        mock_confidence_component.return_value.invoke.return_value = mock_confidence_output

        # Mock the get_memory method
        mock_context.get_memory.return_value = [Message(role="assistant", content="Test memory")]

        # Create and invoke the step
        step = RootCauseStep({"run_id": 1, "step_id": 1})
        step.invoke()

        # Verify that the confidence component was called with the correct parameters
        mock_context.get_memory.assert_called_with("root_cause_analysis")
        mock_confidence_component.return_value.invoke.assert_called_once()

        # Get the confidence request from the mock call
        confidence_request = mock_confidence_component.return_value.invoke.call_args[0][0]
        assert confidence_request.step_goal_description == "root cause analysis"
        assert confidence_request.next_step_goal_description == "figuring out a solution"
        assert confidence_request.run_memory == [Message(role="assistant", content="Test memory")]

        # Verify that the confidence scores and comment were set in the state
        mock_context.state.update.assert_called()
        # Get the context manager from the update call
        context_manager = mock_context.state.update.return_value.__enter__.return_value
        assert context_manager.steps[-1].output_confidence_score == 0.85
        assert context_manager.steps[-1].proceed_confidence_score == 0.75

        assert context_manager.steps[-1].agent_comment_thread is not None
        assert (
            context_manager.steps[-1].agent_comment_thread.messages[0].content
            == "This is a test question"
        )

    @patch("seer.automation.autofix.steps.root_cause_step.AutofixSolutionStep")
    @patch("seer.automation.autofix.steps.root_cause_step.BaseTools")
    @patch("seer.automation.autofix.steps.root_cause_step.RootCauseAnalysisComponent")
    @patch("seer.automation.autofix.steps.root_cause_step.ConfidenceComponent")
    @patch("seer.automation.autofix.steps.root_cause_step.CommentThread")
    def test_confidence_evaluation_no_comment(
        self,
        mock_comment_thread,
        mock_confidence_component,
        mock_root_cause_component,
        mock_base_tools,
        mock_solution_step,
    ):
        mock_context = self._create_base_mock_context()

        # Create a test event
        error_event = next(generate(SentryEventData))

        # Create a proper DefaultStep object
        default_step = DefaultStep(key="default_step", title="Default Step")

        # Create a proper RootCauseStep object
        root_cause_step = RootCauseStepModel(
            key="root_cause_analysis",
            title="Root Cause Analysis",
            type=StepType.ROOT_CAUSE_ANALYSIS,
            causes=[],
        )

        # Set up the state with interactivity enabled
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
                options=AutofixRequestOptions(disable_interactivity=False),  # Enable interactivity
            ),
            steps=[default_step, root_cause_step],  # Add both steps to the steps list
        )

        # Mock the root cause output
        mock_root_cause_output = next(generate(RootCauseAnalysisOutput))
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output

        # Mock the confidence component output with None question
        mock_confidence_output = ConfidenceOutput(
            question=None,
            output_confidence_score=0.95,
            proceed_confidence_score=0.90,  # No question
        )
        mock_confidence_component.return_value.invoke.return_value = mock_confidence_output

        # Mock the get_memory method
        mock_context.get_memory.return_value = [Message(role="assistant", content="Test memory")]

        # Create and invoke the step
        step = RootCauseStep({"run_id": 1, "step_id": 1})
        step.invoke()

        # Verify that the confidence component was called
        mock_context.get_memory.assert_called_with("root_cause_analysis")
        mock_confidence_component.return_value.invoke.assert_called_once()

        # Verify that the confidence scores were set in the state
        mock_context.state.update.assert_called()
        # Get the context manager from the update call
        context_manager = mock_context.state.update.return_value.__enter__.return_value
        assert context_manager.steps[-1].output_confidence_score == 0.95
        assert context_manager.steps[-1].proceed_confidence_score == 0.90

        # Verify that no comment thread was created by checking that CommentThread was not called
        mock_comment_thread.assert_not_called()
