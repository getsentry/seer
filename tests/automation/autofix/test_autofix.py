import json
import unittest
from unittest.mock import MagicMock, patch

from seer.automation.autofix.components.assessment.models import ProblemDiscoveryOutput
from seer.automation.autofix.components.planner.models import PlanningOutput, PlanStep
from seer.automation.autofix.components.retriever import RetrieverOutput
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    RepoDefinition,
    SentryEventData,
)
from seer.automation.models import FileChange, FilePatch
from seer.generator import generate


class TestAutofixPipeline(unittest.TestCase):
    @patch("seer.automation.autofix.autofix_context.AutofixContext")
    @patch("seer.automation.autofix.autofix.ProblemDiscoveryComponent")
    def test_autofix_problem_not_actionable(
        self, mock_problem_discovery_component, mock_autofix_context
    ):
        from seer.automation.autofix.autofix import Autofix

        request: AutofixRequest = next(generate(AutofixRequest))
        request.repos = [next(generate(RepoDefinition))]
        request.issue.events = [next(generate(SentryEventData))]
        autofix = Autofix(mock_autofix_context)

        mock_problem_discovery_component.return_value.invoke.return_value = ProblemDiscoveryOutput(
            description="Test Description", reasoning="Test Reason", actionability_score=0.5
        )

        autofix.invoke(request)

        mock_problem_discovery_component.return_value.invoke.assert_called_once()
        mock_autofix_context.event_manager.send_initial_steps.assert_called_once()
        mock_autofix_context.event_manager.send_problem_discovery_result.assert_called_once()
        mock_autofix_context.event_manager.send_codebase_indexing_repo_check_message.assert_not_called()
        # Assert codebase is NOT created
        assert mock_autofix_context.create_codebase_index.call_count == 0

    @patch("seer.automation.autofix.autofix_context.AutofixContext")
    @patch("seer.automation.autofix.autofix.ProblemDiscoveryComponent")
    @patch("seer.automation.autofix.autofix.PlanningComponent")
    @patch("seer.automation.autofix.autofix.RetrieverComponent")
    @patch("seer.automation.autofix.autofix.ExecutorComponent")
    def test_autofix_happy_path(
        self,
        mock_executor_component,
        mock_retriever_component,
        mock_planning_component,
        mock_problem_discovery_component,
        mock_autofix_context,
    ):
        from seer.automation.autofix.autofix import Autofix

        request: AutofixRequest = next(generate(AutofixRequest))
        request.repos = [next(generate(RepoDefinition)), next(generate(RepoDefinition))]
        autofix = Autofix(mock_autofix_context)

        mock_problem_discovery_component.return_value.invoke.return_value = ProblemDiscoveryOutput(
            description="Test Description", reasoning="Test Reason", actionability_score=0.7
        )
        mock_autofix_context.has_codebase_index.return_value = True
        mock_autofix_context.is_behind.return_value = False
        mock_planning_component.return_value.invoke.return_value = PlanningOutput(
            title="Test Title",
            description="Test Description",
            steps=[
                PlanStep(
                    id=1,
                    title="Test Step",
                    text="Test Text",
                ),
                PlanStep(
                    id=2,
                    title="Test Step",
                    text="Test Text",
                ),
            ],
        )
        mock_retriever_component.return_value.invoke.return_value = next(generate(RetrieverOutput))

        autofix.invoke(request)

        mock_event_manager = mock_autofix_context.event_manager

        mock_event_manager.send_initial_steps.assert_called_once()
        mock_problem_discovery_component.return_value.invoke.assert_called_once()
        mock_event_manager.send_problem_discovery_result.assert_called_once()
        assert mock_event_manager.send_codebase_indexing_repo_check_message.call_count == 2

        # Assert codebase is NOT created
        assert mock_autofix_context.create_codebase_index.call_count == 0

        assert mock_event_manager.send_codebase_indexing_repo_exists_message.call_count == 2
        mock_planning_component.return_value.invoke.assert_called_once()
        mock_event_manager.send_planning_result.assert_called_once()
        assert mock_retriever_component.return_value.invoke.call_count == 2
        assert mock_executor_component.return_value.invoke.call_count == 2
        mock_event_manager.send_autofix_complete.assert_called_once()

    @patch("seer.automation.autofix.autofix_context.AutofixContext")
    @patch("seer.automation.autofix.autofix.ProblemDiscoveryComponent")
    @patch("seer.automation.autofix.autofix.PlanningComponent")
    @patch("seer.automation.autofix.autofix.RetrieverComponent")
    @patch("seer.automation.autofix.autofix.ExecutorComponent")
    def test_autofix_new_index(
        self,
        mock_executor_component,
        mock_retriever_component,
        mock_planning_component,
        mock_problem_discovery_component,
        mock_autofix_context,
    ):
        from seer.automation.autofix.autofix import Autofix

        request: AutofixRequest = next(generate(AutofixRequest))
        request.repos = [next(generate(RepoDefinition)), next(generate(RepoDefinition))]
        autofix = Autofix(mock_autofix_context)

        mock_problem_discovery_component.return_value.invoke.return_value = ProblemDiscoveryOutput(
            description="Test Description", reasoning="Test Reason", actionability_score=0.7
        )
        mock_autofix_context.has_codebase_index.return_value = False
        mock_autofix_context.is_behind.return_value = False
        mock_planning_component.return_value.invoke.return_value = PlanningOutput(
            title="Test Title",
            description="Test Description",
            steps=[
                PlanStep(
                    id=1,
                    title="Test Step",
                    text="Test Text",
                ),
                PlanStep(
                    id=2,
                    title="Test Step",
                    text="Test Text",
                ),
            ],
        )
        mock_retriever_component.return_value.invoke.return_value = next(generate(RetrieverOutput))

        autofix.invoke(request)

        mock_event_manager = mock_autofix_context.event_manager

        mock_event_manager.send_initial_steps.assert_called_once()
        mock_problem_discovery_component.return_value.invoke.assert_called_once()
        mock_event_manager.send_problem_discovery_result.assert_called_once()

        assert mock_event_manager.send_codebase_index_creation_message.call_count == 2
        # Assert codebase is created
        assert mock_autofix_context.create_codebase_index.call_count == 2
        assert mock_event_manager.send_codebase_index_creation_complete_message.call_count == 2

        mock_planning_component.return_value.invoke.assert_called_once()
        mock_event_manager.send_planning_result.assert_called_once()
        assert mock_retriever_component.return_value.invoke.call_count == 2
        assert mock_executor_component.return_value.invoke.call_count == 2
        mock_event_manager.send_autofix_complete.assert_called_once()

    @patch("seer.automation.autofix.autofix_context.AutofixContext")
    @patch("seer.automation.autofix.autofix.ProblemDiscoveryComponent")
    @patch("seer.automation.autofix.autofix.PlanningComponent")
    @patch("seer.automation.autofix.autofix.RetrieverComponent")
    @patch("seer.automation.autofix.autofix.ExecutorComponent")
    def test_autofix_dry_run(
        self,
        mock_executor_component,
        mock_retriever_component,
        mock_planning_component,
        mock_problem_discovery_component,
        mock_autofix_context,
    ):
        from seer.automation.autofix.autofix import Autofix

        # Make it a dry run
        mock_autofix_context.commit_changes = False

        request: AutofixRequest = next(generate(AutofixRequest))
        request.repos = [next(generate(RepoDefinition)), next(generate(RepoDefinition))]
        request.issue.events = [next(generate(SentryEventData))]
        autofix = Autofix(mock_autofix_context)

        mock_problem_discovery_component.return_value.invoke.return_value = ProblemDiscoveryOutput(
            description="Test Description", reasoning="Test Reason", actionability_score=0.7
        )
        mock_autofix_context.has_codebase_index.return_value = True
        mock_autofix_context.is_behind.return_value = False
        mock_planning_component.return_value.invoke.return_value = PlanningOutput(
            title="Test Title",
            description="Test Description",
            steps=[
                PlanStep(
                    id=1,
                    title="Test Step",
                    text="Test Text",
                ),
                PlanStep(
                    id=2,
                    title="Test Step",
                    text="Test Text",
                ),
            ],
        )
        mock_retriever_component.return_value.invoke.return_value = next(generate(RetrieverOutput))

        mock_codebase_index = MagicMock()
        mock_codebase_index.file_changes = [next(generate(FileChange))]
        mock_codebase_index.get_file_patches.return_value = (
            [next(generate(FilePatch))],
            "@@ 1,1 1,1 @@\n-Test\n+Test\n",
        )

        mock_repo_client = mock_codebase_index.repo_client
        mock_repo_client.provider = "github"
        mock_repo_client.repo_owner = "test_owner"
        mock_repo_client.repo_name = "test_repo"

        mock_state = mock_autofix_context.state
        mock_state.get.return_value = next(generate(AutofixContinuation))

        mock_autofix_context.codebases = {
            1: mock_codebase_index,
        }

        autofix_result = autofix.invoke(request)

        mock_event_manager = mock_autofix_context.event_manager

        mock_event_manager.send_initial_steps.assert_called_once()
        mock_problem_discovery_component.return_value.invoke.assert_called_once()
        mock_event_manager.send_problem_discovery_result.assert_called_once()
        assert mock_event_manager.send_codebase_indexing_repo_check_message.call_count == 2

        # Assert codebase is NOT created
        assert mock_autofix_context.create_codebase_index.call_count == 0

        # Make sure the run completes
        assert mock_event_manager.send_codebase_indexing_repo_exists_message.call_count == 2
        mock_planning_component.return_value.invoke.assert_called_once()
        mock_event_manager.send_planning_result.assert_called_once()
        assert mock_retriever_component.return_value.invoke.call_count == 2
        assert mock_executor_component.return_value.invoke.call_count == 2
        mock_event_manager.send_autofix_complete.assert_called_once()

        # Make sure that nothing was committed
        mock_repo_client.create_branch_from_changes.assert_not_called()
        mock_repo_client.create_pr_from_branch.assert_not_called()

        # Make sure diffs are returned
        assert autofix_result is not None
        assert len(autofix_result["outputs"]) == 1
        assert autofix_result["outputs"][0].diff is not None
        assert autofix_result["outputs"][0].diff_str is not None
