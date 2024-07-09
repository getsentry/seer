from unittest.mock import Mock, patch

import pytest
from johen import generate
from langfuse.client import DatasetItemClient

from seer.automation.autofix.evaluations import sync_run_evaluation_on_item
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixRequestOptions,
    AutofixUserDetails,
    ChangesStep,
    CodebaseChange,
    IssueDetails,
    RepoDefinition,
)
from seer.automation.autofix.models import RootCauseStep as RootCauseStepModel
from seer.automation.state import LocalMemoryState


class TestSyncRunEvaluationOnItem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create a more realistic AutofixRequest
        self.autofix_request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            base_commit_sha="abcdef1234567890",
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )

        self.mock_dataset_item = Mock(spec=DatasetItemClient)
        self.mock_dataset_item.input = {"request": self.autofix_request.model_dump()}

        # Create LocalMemoryState
        self.test_state = LocalMemoryState(AutofixContinuation(request=self.autofix_request))

        self.mock_root_cause_step_instance = Mock()
        self.mock_planning_step_instance = Mock()

        # Setup patches
        self.patch_create_initial_run = patch(
            "seer.automation.autofix.evaluations.create_initial_autofix_run"
        )
        self.patch_root_cause_step = patch("seer.automation.autofix.evaluations.RootCauseStep")
        self.patch_event_manager = patch("seer.automation.autofix.evaluations.AutofixEventManager")
        self.patch_planning_step = patch("seer.automation.autofix.evaluations.AutofixPlanningStep")

        # Start patches
        self.mock_create_initial_run = self.patch_create_initial_run.start()
        self.mock_root_cause_step = self.patch_root_cause_step.start()
        self.mock_event_manager = self.patch_event_manager.start()
        self.mock_planning_step = self.patch_planning_step.start()

        # Setup returns
        self.mock_create_initial_run.return_value = self.test_state
        self.mock_root_cause_step.get_signature.return_value = self.mock_root_cause_step_instance
        self.mock_planning_step.get_signature.return_value = self.mock_planning_step_instance

    def teardown_method(self):
        # Stop patches
        self.patch_create_initial_run.stop()
        self.patch_root_cause_step.stop()
        self.patch_event_manager.stop()
        self.patch_planning_step.stop()

    def test_sync_run_evaluation_on_item_happy_path(self):
        # Setup state changes for root cause step
        root_cause_model = next(generate(RootCauseStepModel))
        root_cause_model.causes = [Mock(id=1, suggested_fixes=[Mock(id=2)])]

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        changes_step = next(generate(ChangesStep))
        changes_step.changes = [
            CodebaseChange(
                repo_external_id="1",
                repo_name="test",
                title="123",
                description="123",
                diff=[],
                diff_str="diff str 1",
            ),
            CodebaseChange(
                repo_external_id="1",
                repo_name="test",
                title="123",
                description="123",
                diff=[],
                diff_str="diff str 2",
            ),
        ]

        def planning_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(changes_step)
            return self.test_state

        self.mock_planning_step_instance.apply.side_effect = planning_apply_side_effect

        # Run the function
        result = sync_run_evaluation_on_item(self.mock_dataset_item)

        # Assertions
        assert self.mock_create_initial_run.called
        request_copy = self.autofix_request.model_copy()
        request_copy.options = request_copy.options.model_copy()
        request_copy.options.disable_codebase_indexing = True
        self.mock_create_initial_run.assert_called_once_with(request_copy)

        assert self.mock_root_cause_step.get_signature.called
        assert self.mock_root_cause_step_instance.apply.called

        self.mock_event_manager.return_value.set_selected_root_cause.assert_called_once()

        assert self.mock_planning_step.get_signature.called
        assert self.mock_planning_step_instance.apply.called

        # Check that the state was updated
        final_state = self.test_state.get()
        assert len(final_state.steps) == 2
        assert isinstance(final_state.steps[0], RootCauseStepModel)
        assert isinstance(final_state.steps[1], ChangesStep)

        assert result == "diff str 1\ndiff str 2"
