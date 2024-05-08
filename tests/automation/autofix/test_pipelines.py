import unittest
from unittest.mock import MagicMock, patch

from johen import generate

from seer.automation.autofix.components.planner.models import (
    PlanningOutput,
    PlanningRequest,
    ReplaceCodePromptXml,
)
from seer.automation.autofix.components.retriever import RetrieverOutput
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseAnalysisOutput,
    RootCauseSuggestedFix,
)
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    RootCauseStep,
    SuggestedFixRootCauseSelection,
)
from seer.automation.autofix.pipelines import (
    AutofixExecution,
    AutofixRootCause,
    CheckCodebaseForUpdatesSideEffect,
    CreateAnyMissingCodebaseIndexesSideEffect,
)
from seer.automation.models import EventDetails, IssueDetails, SentryEventData


class TestCreateAnyMissingCodebaseIndexesSideEffect(unittest.TestCase):
    @patch("seer.automation.autofix.pipelines.AutofixContext")
    def test_invoke_creation_not_needed(self, mock_context):
        # Setup
        mock_context.return_value.has_missing_codebase_indexes.return_value = False
        side_effect = CreateAnyMissingCodebaseIndexesSideEffect(mock_context())

        # Test
        side_effect.invoke()

        # Verify
        mock_context.return_value.event_manager.send_codebase_indexing_start.assert_not_called()
        mock_context.return_value.event_manager.create_codebase_index.assert_not_called()

    @patch("seer.automation.autofix.pipelines.AutofixContext")
    def test_invoke_creation_needed(self, mock_context):
        # Setup
        mock_context.return_value.has_missing_codebase_indexes.return_value = True
        mock_context.return_value.repos = [MagicMock()]
        mock_context.return_value.get_codebase_from_external_id.return_value = None
        side_effect = CreateAnyMissingCodebaseIndexesSideEffect(mock_context())

        # Test
        side_effect.invoke()

        # Verify
        mock_context.return_value.event_manager.send_codebase_indexing_start.assert_called_once()
        mock_context.return_value.create_codebase_index.assert_called_once()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_if_exists.assert_called_once()

    @patch("seer.automation.autofix.pipelines.AutofixContext")
    def test_invoke_creation_needed_workspace_not_ready(self, mock_context):
        # Setup
        mock_context.return_value.has_missing_codebase_indexes.return_value = True
        mock_context.return_value.repos = [MagicMock()]
        mock_codebase = MagicMock()
        mock_context.return_value.get_codebase_from_external_id.return_value = mock_codebase

        mock_codebase.workspace.is_ready.return_value = False

        side_effect = CreateAnyMissingCodebaseIndexesSideEffect(mock_context())

        # Test
        side_effect.invoke()

        # Verify
        mock_codebase.workspace.delete.assert_called_once()
        mock_context.return_value.event_manager.send_codebase_indexing_start.assert_called_once()
        mock_context.return_value.create_codebase_index.assert_called_once()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_if_exists.assert_called_once()


class TestCheckCodebaseForUpdatesSideEffect(unittest.TestCase):
    @patch("seer.automation.autofix.pipelines.AutofixContext")
    @patch("seer.automation.autofix.pipelines.update_codebase_index.apply_async")
    @patch("seer.automation.autofix.pipelines.sentry_sdk.start_span")
    def test_invoke_updates_needed(self, mock_start_span, mock_apply_async, mock_context):
        # Setup
        codebases = {
            1: MagicMock(is_behind=MagicMock(return_value=True)),
            2: MagicMock(is_behind=MagicMock(return_value=False)),
        }
        mock_context.return_value.codebases = codebases
        mock_context.return_value.diff_contains_stacktrace_files = MagicMock(return_value=True)
        mock_context.return_value.event_manager = MagicMock()
        error_event = next(generate(SentryEventData))
        mock_context.return_value.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[error_event]),
            )
        )
        mock_context.return_value.has_codebase_indexing_run.return_value = False

        side_effect = CheckCodebaseForUpdatesSideEffect(mock_context())

        # Test
        side_effect.invoke()

        # Verify
        mock_context.return_value.event_manager.send_codebase_indexing_start.assert_called()
        mock_start_span.assert_called()
        codebases[1].update.assert_called()
        codebases[2].update.assert_not_called()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_if_exists.assert_called()

    @patch("seer.automation.autofix.pipelines.AutofixContext")
    @patch("seer.automation.autofix.pipelines.update_codebase_index.apply_async")
    @patch("seer.automation.autofix.pipelines.sentry_sdk.start_span")
    def test_invoke_does_not_run_if_already_indexed(
        self, mock_start_span, mock_apply_async, mock_context
    ):
        # Setup
        codebases = {
            1: MagicMock(is_behind=MagicMock(return_value=True)),
            2: MagicMock(is_behind=MagicMock(return_value=False)),
        }
        mock_context.return_value.codebases = codebases
        mock_context.return_value.diff_contains_stacktrace_files = MagicMock(return_value=True)
        mock_context.return_value.event_manager = MagicMock()
        error_event = next(generate(SentryEventData))
        mock_context.return_value.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[error_event]),
            )
        )
        mock_context.return_value.has_codebase_indexing_run.return_value = True

        side_effect = CheckCodebaseForUpdatesSideEffect(mock_context())

        # Test
        side_effect.invoke()

        # Verify
        mock_context.return_value.event_manager.send_codebase_indexing_start.assert_not_called()
        mock_start_span.assert_not_called()
        codebases[1].update.assert_not_called()
        codebases[2].update.assert_not_called()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_for_repo.assert_not_called()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_if_exists.assert_not_called()

    @patch("seer.automation.autofix.pipelines.update_codebase_index")
    @patch("seer.automation.autofix.pipelines.AutofixContext")
    @patch("seer.automation.autofix.pipelines.update_codebase_index.apply_async")
    @patch("seer.automation.autofix.pipelines.sentry_sdk.start_span")
    def test_invoke_updates_needed_but_can_wait(
        self, mock_start_span, mock_apply_async, mock_context, mock_update_codebase_index
    ):
        # Setup
        codebases = {
            1: MagicMock(is_behind=MagicMock(return_value=True)),
            2: MagicMock(is_behind=MagicMock(return_value=False)),
        }
        mock_context.return_value.codebases = codebases
        mock_context.return_value.diff_contains_stacktrace_files = MagicMock(return_value=False)
        mock_context.return_value.event_manager = MagicMock()
        error_event = next(generate(SentryEventData))
        mock_context.return_value.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[error_event]),
            )
        )
        mock_context.return_value.has_codebase_indexing_run.return_value = False

        side_effect = CheckCodebaseForUpdatesSideEffect(mock_context())

        # Test
        side_effect.invoke()
        print("codebases", codebases)
        # Verify
        mock_context.return_value.event_manager.send_codebase_indexing_start.assert_not_called()
        mock_start_span.assert_not_called()
        codebases[1].update.assert_not_called()
        codebases[2].update.assert_not_called()
        mock_update_codebase_index.apply_async.assert_called()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_for_repo.assert_not_called()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_if_exists.assert_called()

    @patch("seer.automation.autofix.pipelines.AutofixContext")
    @patch("seer.automation.autofix.pipelines.update_codebase_index.apply_async")
    def test_invoke_no_updates_needed(self, mock_apply_async, mock_context):
        # Setup
        codebases = {
            1: MagicMock(is_behind=MagicMock(return_value=False)),
            2: MagicMock(is_behind=MagicMock(return_value=False)),
        }
        mock_context.return_value.codebases = codebases
        mock_context.return_value.diff_contains_stacktrace_files = MagicMock(return_value=False)
        mock_context.return_value.event_manager = MagicMock()
        error_event = next(generate(SentryEventData))
        mock_context.return_value.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[error_event]),
            )
        )
        mock_context.return_value.has_codebase_indexing_run.return_value = False

        side_effect = CheckCodebaseForUpdatesSideEffect(mock_context())

        # Test
        side_effect.invoke()

        # Verify
        mock_apply_async.assert_not_called()
        mock_context.return_value.event_manager.send_codebase_indexing_start.assert_not_called()
        codebases[1].update.assert_not_called()
        codebases[2].update.assert_not_called()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_for_repo.assert_not_called()
        mock_context.return_value.event_manager.send_codebase_indexing_complete_if_exists.assert_called()


class TestRootCausePipeline(unittest.TestCase):
    @patch("seer.automation.autofix.pipelines.RootCauseAnalysisComponent")
    @patch("seer.automation.autofix.pipelines.CheckCodebaseForUpdatesSideEffect")
    @patch("seer.automation.autofix.pipelines.AutofixContext")
    def test_invoke(self, mock_context, mock_update_side_effect, mock_root_cause_component):
        error_event = next(generate(SentryEventData))

        mock_context.return_value.has_missing_codebase_indexes.return_value = False
        mock_context.return_value.state.get.return_value = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[error_event]),
            )
        )

        mock_root_cause_output = next(generate(RootCauseAnalysisOutput))
        mock_root_cause_component.return_value.invoke.return_value = mock_root_cause_output

        root_cause_pipeline = AutofixRootCause(mock_context())
        root_cause_pipeline.invoke()

        mock_context.return_value.event_manager.send_root_cause_analysis_start.assert_called_once()
        mock_update_side_effect.return_value.invoke.assert_called_once()
        mock_context.return_value.event_manager.send_root_cause_analysis_result.assert_called_once()
        mock_context.return_value.event_manager.send_root_cause_analysis_result.assert_called_with(
            mock_root_cause_output
        )


class TestAutofixExecutionPipeline(unittest.TestCase):
    @patch("seer.automation.autofix.pipelines.PlanningComponent")
    @patch("seer.automation.autofix.pipelines.ExecutorComponent")
    @patch("seer.automation.autofix.pipelines.RetrieverComponent")
    @patch("seer.automation.autofix.pipelines.AutofixContext")
    def test_invoke(self, mock_context, mock_retriever, mock_executor, mock_planning):
        # Setup mock context with necessary state
        mock_root_cause = next(generate(RootCauseAnalysisItem))
        mock_root_cause.suggested_fixes = [next(generate(RootCauseSuggestedFix))]
        mock_state = AutofixContinuation(
            request=AutofixRequest(
                organization_id=1,
                project_id=1,
                repos=[],
                issue=IssueDetails(id=0, title="", events=[next(generate(SentryEventData))]),
                instruction="instruction",
            ),
            steps=[
                RootCauseStep(
                    id="root_cause_analysis",
                    title="Root Cause Analysis",
                    causes=[mock_root_cause],
                    selection=SuggestedFixRootCauseSelection(
                        cause_id=mock_root_cause.id, fix_id=mock_root_cause.suggested_fixes[0].id
                    ),
                )
            ],
        )
        mock_context.return_value.state.get.return_value = mock_state
        mock_context.return_value.has_missing_codebase_indexes.return_value = False

        # Setup PlanningComponent to return a planning output
        mock_planning_output = PlanningOutput(
            tasks=[
                ReplaceCodePromptXml(
                    file_path="",
                    repo_name="",
                    reference_snippet="",
                    new_snippet="",
                    new_imports="",
                    description="",
                    commit_message="",
                )
            ],
        )
        mock_planning.return_value.invoke.return_value = mock_planning_output

        mock_retriever.return_value.invoke.return_value = next(generate(RetrieverOutput))

        # Setup ExecutorComponent to simulate execution
        mock_executor.return_value.invoke.return_value = None

        # Create instance of AutofixExecution
        execution_pipeline = AutofixExecution(mock_context())

        # Invoke the execution pipeline
        execution_pipeline.invoke()

        # Assertions to ensure all components are called correctly
        mock_context.return_value.event_manager.send_planning_start.assert_called_once()
        mock_planning.return_value.invoke.assert_called_once()
        mock_planning.return_value.invoke.assert_called_with(
            PlanningRequest(
                event_details=EventDetails.from_event(mock_state.request.issue.events[0]),
                root_cause_and_fix=mock_root_cause,
                instruction="instruction",
            )
        )
        mock_executor.return_value.invoke.assert_called()
        mock_retriever.return_value.invoke.assert_called()
        mock_context.return_value.event_manager.send_planning_result.assert_called_with(
            mock_planning_output
        )
        mock_context.return_value.event_manager.send_execution_complete.assert_called()
