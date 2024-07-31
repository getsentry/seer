from unittest.mock import Mock, patch

import pytest
from johen import generate
from langfuse.client import DatasetItemClient

from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseRelevantContext,
)
from seer.automation.autofix.evaluations import (
    score_fix_single_it,
    score_one,
    score_root_cause_single_it,
    score_root_causes,
    sync_run_evaluation_on_item,
    sync_run_root_cause,
)
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
from seer.automation.models import SentryEventData
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
        self.patch_planning_step = patch("seer.automation.autofix.evaluations.AutofixCodingStep")

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
        root_cause_model.causes = [Mock(id=1, code_context=[Mock(id=2)])]

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

    def test_sync_run_evaluation_on_item_no_root_causes(self):
        # Setup state changes for root cause step with no causes
        root_cause_model = RootCauseStepModel(
            id="root_cause_analysis", title="Root Cause Analysis", causes=[]
        )

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        # Run the function
        result = sync_run_evaluation_on_item(self.mock_dataset_item)

        # Assertions
        assert result is None
        assert not self.mock_planning_step.get_signature.called

    def test_sync_run_evaluation_on_item_no_code_context(self):
        # Setup state changes for root cause step with causes but no code context
        root_cause_model = RootCauseStepModel(
            id="root_cause_analysis",
            title="Root Cause Analysis",
            causes=[
                RootCauseAnalysisItem(
                    id=1,
                    title="Test Cause",
                    description="Test cause description",
                    likelihood=0.8,
                    actionability=0.7,
                    code_context=[],
                )
            ],
        )

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        # Run the function
        result = sync_run_evaluation_on_item(self.mock_dataset_item)

        # Assertions
        assert result is None
        assert not self.mock_planning_step.get_signature.called

    def test_sync_run_evaluation_on_item_no_changes(self):
        # Setup state changes for root cause step
        root_cause_model = RootCauseStepModel(
            id="root_cause_analysis",
            title="Root Cause Analysis",
            causes=[
                RootCauseAnalysisItem(
                    id=1,
                    title="Test Cause",
                    description="Test cause description",
                    likelihood=0.8,
                    actionability=0.7,
                    code_context=[
                        RootCauseRelevantContext(
                            id=2,
                            title="Test Fix",
                            description="Test fix description",
                        )
                    ],
                )
            ],
        )

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        # Setup state changes for planning step with no changes
        changes_step = ChangesStep(id="changes", title="Changes", changes=[])

        def planning_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(changes_step)
            return self.test_state

        self.mock_planning_step_instance.apply.side_effect = planning_apply_side_effect

        # Run the function
        result = sync_run_evaluation_on_item(self.mock_dataset_item)

        # Assertions
        assert result is None
        assert self.mock_planning_step.get_signature.called
        assert self.mock_planning_step_instance.apply.called


class TestSyncRunRootCause:
    @pytest.fixture(autouse=True)
    def setup(self):
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

        self.test_state = LocalMemoryState(AutofixContinuation(request=self.autofix_request))

        self.mock_root_cause_step_instance = Mock()

        self.patch_create_initial_run = patch(
            "seer.automation.autofix.evaluations.create_initial_autofix_run"
        )
        self.patch_root_cause_step = patch("seer.automation.autofix.evaluations.RootCauseStep")

        self.mock_create_initial_run = self.patch_create_initial_run.start()
        self.mock_root_cause_step = self.patch_root_cause_step.start()

        self.mock_create_initial_run.return_value = self.test_state
        self.mock_root_cause_step.get_signature.return_value = self.mock_root_cause_step_instance

    def teardown_method(self):
        self.patch_create_initial_run.stop()
        self.patch_root_cause_step.stop()

    def test_sync_run_root_cause_happy_path(self):
        root_cause_model = next(generate(RootCauseStepModel))
        root_cause_model.causes = [Mock(id=1, code_context=[Mock(id=2)])]

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        result = sync_run_root_cause(self.mock_dataset_item)

        assert self.mock_create_initial_run.called
        assert self.mock_root_cause_step.get_signature.called
        assert self.mock_root_cause_step_instance.apply.called
        assert result == root_cause_model.causes

    def test_sync_run_root_cause_no_causes(self):
        root_cause_model = RootCauseStepModel(
            id="root_cause_analysis", title="Root Cause Analysis", causes=[]
        )

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        with pytest.raises(ValueError, match="Expected root cause step"):
            sync_run_root_cause(self.mock_dataset_item)


class TestScoreRootCauseSingleIt:
    @pytest.fixture
    def mock_dataset_item(self):
        mock_item = Mock(spec=DatasetItemClient)
        mock_item.expected_output = {
            "root_cause": "Expected root cause",
            "solution_summary": "Expected solution summary",
            "diff": {"file_path": "test.py", "code_diff": "expected diff"},
        }
        mock_item.input = {
            "request": AutofixRequest(
                organization_id=1,
                project_id=2,
                repos=[
                    RepoDefinition(
                        provider="github", owner="test", name="test-repo", external_id="1"
                    )
                ],
                issue=IssueDetails(
                    id=1, title="Test Issue", short_id="1", events=[next(generate(SentryEventData))]
                ),
                invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
                base_commit_sha="abcdef1234567890",
                instruction="Fix the bug",
                options=AutofixRequestOptions(),
            ).model_dump()
        }
        return mock_item

    @patch("seer.automation.autofix.evaluations.GptClient")
    def test_score_root_cause_single_it(self, mock_gpt_client, mock_dataset_item):
        mock_gpt_instance = Mock()
        mock_gpt_client.return_value = mock_gpt_instance
        mock_gpt_instance.completion.return_value = (
            Mock(content="<score_1>0.8</score_1><score_2>0.6</score_2>"),
            None,
        )

        causes = [
            RootCauseAnalysisItem(
                id=1,
                title="Cause 1",
                description="Description 1",
                likelihood=0.8,
                actionability=0.7,
            ),
            RootCauseAnalysisItem(
                id=2,
                title="Cause 2",
                description="Description 2",
                likelihood=0.6,
                actionability=0.5,
            ),
        ]

        result = score_root_cause_single_it(mock_dataset_item, causes, model="test_model")

        assert result == [0.8, 0.6]
        mock_gpt_instance.completion.assert_called_once()

    @patch("seer.automation.autofix.evaluations.GptClient")
    def test_score_root_cause_single_it_no_score(self, mock_gpt_client, mock_dataset_item):
        mock_gpt_instance = Mock()
        mock_gpt_client.return_value = mock_gpt_instance
        mock_gpt_instance.completion.return_value = (Mock(content="No score provided"), None)

        causes = [
            RootCauseAnalysisItem(
                id=1,
                title="Cause 1",
                description="Description 1",
                likelihood=0.8,
                actionability=0.7,
            )
        ]

        result = score_root_cause_single_it(mock_dataset_item, causes, model="test_model")

        assert result == [0]

    def test_score_root_cause_single_it_missing_expected_output(self, mock_dataset_item):
        mock_dataset_item.expected_output = None

        causes = [
            RootCauseAnalysisItem(
                id=1,
                title="Cause 1",
                description="Description 1",
                likelihood=0.8,
                actionability=0.7,
            )
        ]

        with pytest.raises(ValueError, match="Expected output is missing from dataset item"):
            score_root_cause_single_it(mock_dataset_item, causes, model="test_model")


class TestScoreRootCauses:
    @pytest.fixture
    def mock_dataset_item(self):
        return Mock(spec=DatasetItemClient)

    @patch("seer.automation.autofix.evaluations.score_root_cause_single_it")
    def test_score_root_causes(self, mock_score_root_cause_single_it, mock_dataset_item):
        mock_score_root_cause_single_it.side_effect = [[0.8, 0.6], [0.7, 0.5], [0.9, 0.7]]

        causes = [
            RootCauseAnalysisItem(
                id=1,
                title="Cause 1",
                description="Description 1",
                likelihood=0.8,
                actionability=0.7,
            ),
            RootCauseAnalysisItem(
                id=2,
                title="Cause 2",
                description="Description 2",
                likelihood=0.6,
                actionability=0.5,
            ),
        ]

        result = score_root_causes(mock_dataset_item, causes, n_panel=3, model="test_model")

        assert result == {"highest_score": 0.8, "position_score": 1.0, "mean_score": 0.7}
        assert mock_score_root_cause_single_it.call_count == 3

    @patch("seer.automation.autofix.evaluations.score_root_cause_single_it")
    def test_score_root_causes_custom_n_panel(
        self, mock_score_root_cause_single_it, mock_dataset_item
    ):
        mock_score_root_cause_single_it.side_effect = [[0.8, 0.6], [0.7, 0.5]]

        causes = [
            RootCauseAnalysisItem(
                id=1,
                title="Cause 1",
                description="Description 1",
                likelihood=0.8,
                actionability=0.7,
            ),
            RootCauseAnalysisItem(
                id=2,
                title="Cause 2",
                description="Description 2",
                likelihood=0.6,
                actionability=0.5,
            ),
        ]

        result = score_root_causes(mock_dataset_item, causes, n_panel=2, model="test_model")

        assert result == {"highest_score": 0.75, "position_score": 1.0, "mean_score": 0.65}
        assert mock_score_root_cause_single_it.call_count == 2


class TestScoreFixSingleIt:
    @pytest.fixture
    def mock_dataset_item(self):
        mock_item = Mock(spec=DatasetItemClient)
        mock_item.expected_output = {"diff": "expected diff"}
        mock_item.input = {
            "request": AutofixRequest(
                organization_id=1,
                project_id=2,
                repos=[
                    RepoDefinition(
                        provider="github", owner="test", name="test-repo", external_id="1"
                    )
                ],
                issue=IssueDetails(
                    id=1, title="Test Issue", short_id="1", events=[next(generate(SentryEventData))]
                ),
                invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
                base_commit_sha="abcdef1234567890",
                instruction="Fix the bug",
                options=AutofixRequestOptions(),
            ).model_dump()
        }
        return mock_item

    @patch("seer.automation.autofix.evaluations.GptClient")
    def test_score_fix_single_it(self, mock_gpt_client, mock_dataset_item):
        mock_gpt_instance = Mock()
        mock_gpt_client.return_value = mock_gpt_instance
        mock_gpt_instance.completion.return_value = (Mock(content="<score>0.8</score>"), None)

        result = score_fix_single_it(mock_dataset_item, "predicted diff", model="test_model")

        assert result == 0.8
        mock_gpt_instance.completion.assert_called_once()

    @patch("seer.automation.autofix.evaluations.GptClient")
    def test_score_fix_single_it_no_score(self, mock_gpt_client, mock_dataset_item):
        mock_gpt_instance = Mock()
        mock_gpt_client.return_value = mock_gpt_instance
        mock_gpt_instance.completion.return_value = (Mock(content="No score provided"), None)

        result = score_fix_single_it(mock_dataset_item, "predicted diff", model="test_model")

        assert result == 0

    def test_score_fix_single_it_missing_expected_output(self, mock_dataset_item):
        mock_dataset_item.expected_output = None

        with pytest.raises(ValueError, match="Expected output is missing from dataset item"):
            score_fix_single_it(mock_dataset_item, "predicted diff", model="test_model")


class TestScoreOne:
    @pytest.fixture
    def mock_dataset_item(self):
        return Mock(spec=DatasetItemClient)

    @patch("seer.automation.autofix.evaluations.score_fix_single_it")
    def test_score_one(self, mock_score_fix_single_it, mock_dataset_item):
        mock_score_fix_single_it.side_effect = [0.7, 0.8, 0.9]

        result = score_one(mock_dataset_item, "predicted diff", n_panel=3, model="test_model")

        assert result == 0.8
        assert mock_score_fix_single_it.call_count == 3

    @patch("seer.automation.autofix.evaluations.score_fix_single_it")
    def test_score_one_custom_n_panel(self, mock_score_fix_single_it, mock_dataset_item):
        mock_score_fix_single_it.side_effect = [0.6, 0.8]

        result = score_one(mock_dataset_item, "predicted diff", n_panel=2, model="test_model")

        assert result == 0.7
        assert mock_score_fix_single_it.call_count == 2
