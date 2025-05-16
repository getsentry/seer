from unittest.mock import Mock, PropertyMock, patch

import pytest
from johen import generate
from langfuse.client import DatasetItemClient

from seer.automation.agent.models import (
    LlmGenerateTextResponse,
    LlmProviderType,
    LlmResponseMetadata,
    Message,
    Usage,
)
from seer.automation.autofix.components.root_cause.models import (
    RelevantCodeFile,
    RootCauseAnalysisItem,
    TimelineEvent,
)
from seer.automation.autofix.components.solution.models import SolutionTimelineEvent
from seer.automation.autofix.evaluations import (
    score_coding,
    score_coding_single_it,
    score_root_cause_single_it,
    score_root_causes,
    score_solution,
    score_solution_single_it,
    sync_run_evaluation_on_item,
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
from seer.automation.autofix.models import SolutionStep
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
        root_cause_model.key = "root_cause_analysis"
        root_cause_model.causes = [
            RootCauseAnalysisItem(
                id=1,
                root_cause_reproduction=[
                    TimelineEvent(
                        title="Test Cause",
                        code_snippet_and_analysis="The root cause of the issue is ...",
                        timeline_item_type="internal_code",
                        relevant_code_file=RelevantCodeFile(
                            file_path="test.py",
                            repo_name="owner/repo",
                        ),
                        is_most_important_event=True,
                    )
                ],
            )
        ]

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        changes_step = next(generate(ChangesStep))
        changes_step.key = "changes"
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
        result_state = sync_run_evaluation_on_item(self.mock_dataset_item)

        # Assertions
        assert self.mock_create_initial_run.called
        request_copy = self.autofix_request.model_copy()
        request_copy.options = request_copy.options.model_copy()
        request_copy.options.disable_codebase_indexing = True
        request_copy.options.disable_interactivity = True
        request_copy.options.force_use_repos = True
        self.mock_create_initial_run.assert_called_once_with(request_copy)

        assert self.mock_root_cause_step.get_signature.called
        assert self.mock_root_cause_step_instance.apply.called

        assert self.mock_planning_step.get_signature.called
        assert self.mock_planning_step_instance.apply.called

        # Check that the state was updated
        assert len(result_state.steps) == 2
        assert isinstance(result_state.steps[0], RootCauseStepModel)
        assert isinstance(result_state.steps[1], ChangesStep)
        assert result_state.root_cause_step is not None
        assert result_state.changes_step is not None
        assert result_state.changes_step.changes[0].diff_str == "diff str 1"
        assert result_state.changes_step.changes[1].diff_str == "diff str 2"

    def test_sync_run_evaluation_on_item_no_root_causes(self):
        # Setup state changes for root cause step with no causes
        root_cause_model = RootCauseStepModel(
            id="root_cause_analysis",
            key="root_cause_analysis",
            title="Root Cause Analysis",
            causes=[],
        )

        def root_cause_apply_side_effect():
            with self.test_state.update() as cur:
                cur.steps.append(root_cause_model)
            return self.test_state

        self.mock_root_cause_step_instance.apply.side_effect = root_cause_apply_side_effect

        # Run the function
        result_state = sync_run_evaluation_on_item(self.mock_dataset_item)

        # Assertions
        assert result_state.root_cause_step is not None
        assert not result_state.root_cause_step.causes
        assert not self.mock_planning_step.get_signature.called

    def test_sync_run_evaluation_on_item_exception(self):
        # Setup mock to raise exception
        self.mock_root_cause_step_instance.apply.side_effect = Exception("Test exception")

        # We need to patch the AutofixEventManager to return our empty state
        with patch(
            "seer.automation.autofix.evaluations.create_initial_autofix_run",
            return_value=self.test_state,
        ):
            # Run the function - should not raise the exception
            result_state = sync_run_evaluation_on_item(self.mock_dataset_item)

            # Assertions
            assert result_state is not None
            # In the actual implementation, we catch the exception and return state.get()
            # So we're testing that it returns a state regardless of the exception


class TestScoreRootCauseSingleIt:
    @pytest.fixture
    def mock_dataset_item(self):
        mock_item = Mock(spec=DatasetItemClient)
        mock_item.expected_output = {
            "root_cause": "Expected root cause",
            "solution_summary": "Expected solution summary",
            "diff": {"file_path": "test.py", "code_diff": "expected diff"},
            "solution_diff": {
                "description": "expected solution diff",
                "unified_diff": "expected solution diff",
            },
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
                instruction="Fix the bug",
                options=AutofixRequestOptions(),
            ).model_dump()
        }
        return mock_item

    @pytest.fixture
    def mock_final_state(self):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        root_cause_step = RootCauseStepModel(
            id="root_cause_analysis",
            key="root_cause_analysis",
            title="Root Cause Analysis",
            causes=[
                RootCauseAnalysisItem(
                    id=1,
                    root_cause_reproduction=[
                        TimelineEvent(
                            title="Test Cause",
                            code_snippet_and_analysis="The root cause of the issue is ...",
                            timeline_item_type="internal_code",
                            relevant_code_file=RelevantCodeFile(
                                file_path="test.py",
                                repo_name="owner/repo",
                            ),
                            is_most_important_event=True,
                        )
                    ],
                )
            ],
        )
        final_state.steps.append(root_cause_step)
        return final_state

    @patch("seer.automation.autofix.evaluations.LlmClient")
    def test_score_root_cause_single_it(self, mock_llm_client, mock_dataset_item, mock_final_state):
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.generate_text.return_value = LlmGenerateTextResponse(
            message=Message(
                role="assistant",
                content="<score>0.8</score><verdict>True</verdict><helpful>True</helpful>",
            ),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = score_root_cause_single_it(mock_dataset_item, mock_final_state, model="test_model")

        assert result == (0.8, True, True)
        mock_llm_instance.generate_text.assert_called_once()

    @patch("seer.automation.autofix.evaluations.LlmClient")
    def test_score_root_cause_single_it_no_score(
        self, mock_llm_client, mock_dataset_item, mock_final_state
    ):
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.generate_text.return_value = LlmGenerateTextResponse(
            message=Message(role="assistant", content="No score provided"),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = score_root_cause_single_it(mock_dataset_item, mock_final_state, model="test_model")

        assert result == (0, False, False)

    def test_score_root_cause_single_it_missing_expected_output(
        self, mock_dataset_item, mock_final_state
    ):
        mock_dataset_item.expected_output = None

        with pytest.raises(ValueError, match="Expected output is missing from dataset item"):
            score_root_cause_single_it(mock_dataset_item, mock_final_state, model="test_model")

    def test_score_root_cause_single_it_no_root_cause_step(self, mock_dataset_item):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        result = score_root_cause_single_it(mock_dataset_item, final_state, model="test_model")
        assert result is None


class TestScoreRootCauses:
    @pytest.fixture
    def mock_dataset_item(self):
        return Mock(spec=DatasetItemClient)

    @pytest.fixture
    def mock_final_state(self):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        root_cause_step = RootCauseStepModel(
            id="root_cause_analysis",
            key="root_cause_analysis",
            title="Root Cause Analysis",
            causes=[
                RootCauseAnalysisItem(
                    id=1,
                    root_cause_reproduction=[
                        TimelineEvent(
                            title="Test Cause",
                            code_snippet_and_analysis="The root cause of the issue is ...",
                            timeline_item_type="internal_code",
                            relevant_code_file=RelevantCodeFile(
                                file_path="test.py",
                                repo_name="owner/repo",
                            ),
                            is_most_important_event=True,
                        )
                    ],
                )
            ],
        )
        final_state.steps.append(root_cause_step)
        return final_state

    @patch("seer.automation.autofix.evaluations.score_root_cause_single_it")
    def test_score_root_causes(
        self, mock_score_root_cause_single_it, mock_dataset_item, mock_final_state
    ):
        mock_score_root_cause_single_it.side_effect = [
            (0.8, True, True),
            (0.7, True, True),
            (0.9, True, True),
        ]

        result = score_root_causes(
            mock_dataset_item, mock_final_state, n_panel=3, model="test_model"
        )

        assert result == (0.8, True, True)  # Average score and majority verdict
        assert mock_score_root_cause_single_it.call_count == 3

    @patch("seer.automation.autofix.evaluations.score_root_cause_single_it")
    def test_score_root_causes_custom_n_panel(
        self, mock_score_root_cause_single_it, mock_dataset_item, mock_final_state
    ):
        # Changed side effect to have majority True (2 True, 0 False)
        mock_score_root_cause_single_it.side_effect = [
            (0.6, True, True),
            (0.8, True, True),
        ]

        result = score_root_causes(
            mock_dataset_item, mock_final_state, n_panel=2, model="test_model"
        )

        assert result == (0.7, True, True)  # Average score and majority verdict (2 True > 0 False)
        assert mock_score_root_cause_single_it.call_count == 2

    @patch("seer.automation.autofix.evaluations.score_root_cause_single_it")
    def test_score_root_causes_no_root_cause_step(
        self, mock_score_root_cause_single_it, mock_dataset_item
    ):
        # Set the mock to return None to simulate no root cause
        mock_score_root_cause_single_it.return_value = None

        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)

        # Patch the root_cause_step property to always return None
        with patch.object(
            AutofixContinuation, "root_cause_step", new_callable=PropertyMock, return_value=None
        ):
            result = score_root_causes(
                mock_dataset_item, final_state, n_panel=3, model="test_model"
            )
            assert result is None


class TestScoreSolutionSingleIt:
    @pytest.fixture
    def mock_dataset_item(self):
        mock_item = Mock(spec=DatasetItemClient)
        mock_item.expected_output = {
            "original_diff": "expected diff",
            "solution_diff": {
                "description": "expected solution diff",
                "unified_diff": "expected solution diff",
            },
            "root_cause": "expected root cause",
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
                instruction="Fix the bug",
                options=AutofixRequestOptions(),
            ).model_dump()
        }
        return mock_item

    @pytest.fixture
    def mock_final_state(self):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        solution_step = SolutionStep(
            id="solution",
            key="solution",
            title="Solution",
            solution=[
                SolutionTimelineEvent(
                    title="Test Solution",
                    code_snippet_and_analysis="This is a test solution",
                    relevant_code_file=None,
                )
            ],
        )
        final_state.steps.append(solution_step)
        return final_state

    @patch("seer.automation.autofix.evaluations.LlmClient")
    def test_score_solution_single_it(self, mock_llm_client, mock_dataset_item, mock_final_state):
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.generate_text.return_value = LlmGenerateTextResponse(
            message=Message(role="assistant", content="<score>0.8</score><verdict>True</verdict>"),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = score_solution_single_it(mock_dataset_item, mock_final_state, model="test_model")

        assert result == (0.8, True)
        mock_llm_instance.generate_text.assert_called_once()

    @patch("seer.automation.autofix.evaluations.LlmClient")
    def test_score_solution_single_it_no_score(
        self, mock_llm_client, mock_dataset_item, mock_final_state
    ):
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.generate_text.return_value = LlmGenerateTextResponse(
            message=Message(role="assistant", content="No score provided"),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = score_solution_single_it(mock_dataset_item, mock_final_state, model="test_model")

        assert result == (0, False)

    def test_score_solution_single_it_missing_expected_output(
        self, mock_dataset_item, mock_final_state
    ):
        mock_dataset_item.expected_output = None

        with pytest.raises(ValueError, match="Expected output is missing from dataset item"):
            score_solution_single_it(mock_dataset_item, mock_final_state, model="test_model")

    def test_score_solution_single_it_no_solution_step(self, mock_dataset_item):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        result = score_solution_single_it(mock_dataset_item, final_state, model="test_model")
        assert result is None


class TestScoreSolution:
    @pytest.fixture
    def mock_dataset_item(self):
        return Mock(spec=DatasetItemClient)

    @pytest.fixture
    def mock_final_state(self):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        solution_step = SolutionStep(
            id="solution",
            key="solution",
            title="Solution",
            solution=[
                SolutionTimelineEvent(
                    title="Test Solution",
                    code_snippet_and_analysis="This is a test solution",
                    relevant_code_file=None,
                )
            ],
        )
        final_state.steps.append(solution_step)
        return final_state

    @patch("seer.automation.autofix.evaluations.score_solution_single_it")
    def test_score_solution(
        self, mock_score_solution_single_it, mock_dataset_item, mock_final_state
    ):
        mock_score_solution_single_it.side_effect = [(0.7, True), (0.8, True), (0.9, True)]

        result = score_solution(mock_dataset_item, mock_final_state, n_panel=3, model="test_model")

        assert result == (0.8, True)  # Average score and majority verdict
        assert mock_score_solution_single_it.call_count == 3

    @patch("seer.automation.autofix.evaluations.score_solution_single_it")
    def test_score_solution_custom_n_panel(
        self, mock_score_solution_single_it, mock_dataset_item, mock_final_state
    ):
        # Changed side effect to have majority True (2 True, 0 False)
        mock_score_solution_single_it.side_effect = [(0.6, True), (0.8, True)]

        result = score_solution(mock_dataset_item, mock_final_state, n_panel=2, model="test_model")

        assert result == (0.7, True)  # Average score and majority verdict (2 True > 0 False)
        assert mock_score_solution_single_it.call_count == 2

    @patch("seer.automation.autofix.evaluations.score_solution_single_it")
    def test_score_solution_no_solution_step(
        self, mock_score_solution_single_it, mock_dataset_item
    ):
        # Set the mock to return None to simulate no solution
        mock_score_solution_single_it.return_value = None

        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)

        # Patch the solution_step property to always return None
        with patch.object(
            AutofixContinuation, "solution_step", new_callable=PropertyMock, return_value=None
        ):
            result = score_solution(mock_dataset_item, final_state, n_panel=3, model="test_model")
            assert result is None


class TestScoreCodingSingleIt:
    @pytest.fixture
    def mock_dataset_item(self):
        mock_item = Mock(spec=DatasetItemClient)
        mock_item.expected_output = {
            "original_diff": "expected diff",
            "solution_diff": {
                "description": "expected solution diff",
                "unified_diff": "expected solution diff",
            },
            "root_cause": "expected root cause",
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
                instruction="Fix the bug",
                options=AutofixRequestOptions(),
            ).model_dump()
        }
        return mock_item

    @pytest.fixture
    def mock_final_state(self):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)

        # Add solution step
        solution_step = SolutionStep(
            id="solution",
            key="solution",
            title="Solution",
            solution=[
                SolutionTimelineEvent(
                    title="Test Solution",
                    code_snippet_and_analysis="This is a test solution",
                    relevant_code_file=None,
                )
            ],
        )
        final_state.steps.append(solution_step)

        # Add changes step
        changes_step = ChangesStep(
            id="changes",
            key="changes",
            title="Changes",
            changes=[
                CodebaseChange(
                    repo_external_id="1",
                    repo_name="test",
                    title="Test Change",
                    description="This is a test change",
                    diff=[],
                    diff_str="diff --git a/test.py b/test.py\nindex 123..456 789\n--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-old code\n+new code",
                )
            ],
        )
        final_state.steps.append(changes_step)

        return final_state

    @patch("seer.automation.autofix.evaluations.LlmClient")
    def test_score_coding_single_it(self, mock_llm_client, mock_dataset_item, mock_final_state):
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.generate_text.return_value = LlmGenerateTextResponse(
            message=Message(
                role="assistant",
                content="<correctness_score>0.8</correctness_score><conciseness_score>0.9</conciseness_score>",
            ),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = score_coding_single_it(mock_dataset_item, mock_final_state, model="test_model")

        assert result == (0.8, 0.9)
        mock_llm_instance.generate_text.assert_called_once()

    @patch("seer.automation.autofix.evaluations.LlmClient")
    def test_score_coding_single_it_no_score(
        self, mock_llm_client, mock_dataset_item, mock_final_state
    ):
        mock_llm_instance = Mock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.generate_text.return_value = LlmGenerateTextResponse(
            message=Message(role="assistant", content="No score provided"),
            metadata=LlmResponseMetadata(
                model="test_model",
                provider_name=LlmProviderType.OPENAI,
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        )

        result = score_coding_single_it(mock_dataset_item, mock_final_state, model="test_model")

        assert result == (0, 0)

    def test_score_coding_single_it_no_solution_step(self, mock_dataset_item):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        result = score_coding_single_it(mock_dataset_item, final_state, model="test_model")
        assert result is None

    def test_score_coding_single_it_no_changes_step(self, mock_dataset_item):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)
        solution_step = SolutionStep(
            id="solution",
            key="solution",
            title="Solution",
            solution=[
                SolutionTimelineEvent(
                    title="Test Solution",
                    code_snippet_and_analysis="This is a test solution",
                    relevant_code_file=None,
                )
            ],
        )
        final_state.steps.append(solution_step)
        result = score_coding_single_it(mock_dataset_item, final_state, model="test_model")
        assert result is None


class TestScoreCoding:
    @pytest.fixture
    def mock_dataset_item(self):
        return Mock(spec=DatasetItemClient)

    @pytest.fixture
    def mock_final_state(self):
        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)

        # Add solution step
        solution_step = SolutionStep(
            id="solution",
            key="solution",
            title="Solution",
            solution=[
                SolutionTimelineEvent(
                    title="Test Solution",
                    code_snippet_and_analysis="This is a test solution",
                    relevant_code_file=None,
                )
            ],
        )
        final_state.steps.append(solution_step)

        # Add changes step
        changes_step = ChangesStep(
            id="changes",
            key="changes",
            title="Changes",
            changes=[
                CodebaseChange(
                    repo_external_id="1",
                    repo_name="test",
                    title="Test Change",
                    description="This is a test change",
                    diff=[],
                    diff_str="diff --git a/test.py b/test.py\nindex 123..456 789\n--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-old code\n+new code",
                )
            ],
        )
        final_state.steps.append(changes_step)

        return final_state

    @patch("seer.automation.autofix.evaluations.score_coding_single_it")
    def test_score_coding(self, mock_score_coding_single_it, mock_dataset_item, mock_final_state):
        mock_score_coding_single_it.side_effect = [
            (0.7, 0.8),
            (0.8, 0.7),
            (0.9, 0.9),
        ]

        result = score_coding(mock_dataset_item, mock_final_state, n_panel=3, model="test_model")

        assert result == (0.8, 0.8)  # Average scores
        assert mock_score_coding_single_it.call_count == 3

    @patch("seer.automation.autofix.evaluations.score_coding_single_it")
    def test_score_coding_custom_n_panel(
        self, mock_score_coding_single_it, mock_dataset_item, mock_final_state
    ):
        mock_score_coding_single_it.side_effect = [(0.6, 0.9), (0.8, 0.7)]

        result = score_coding(mock_dataset_item, mock_final_state, n_panel=2, model="test_model")

        assert result == (0.7, 0.8)  # Average scores
        assert mock_score_coding_single_it.call_count == 2

    @patch("seer.automation.autofix.evaluations.score_coding_single_it")
    def test_score_coding_no_solution_step(self, mock_score_coding_single_it, mock_dataset_item):
        # Set the mock to return None to simulate no solution
        mock_score_coding_single_it.return_value = None

        request = AutofixRequest(
            organization_id=1,
            project_id=2,
            repos=[
                RepoDefinition(provider="github", owner="test", name="test-repo", external_id="1")
            ],
            issue=IssueDetails(id=1, title="Test Issue", short_id="1", events=[]),
            invoking_user=AutofixUserDetails(id=1, display_name="Test User"),
            instruction="Fix the bug",
            options=AutofixRequestOptions(),
        )
        final_state = AutofixContinuation(request=request)

        # Patch the solution_step property to always return None
        with patch.object(
            AutofixContinuation, "solution_step", new_callable=PropertyMock, return_value=None
        ):
            result = score_coding(mock_dataset_item, final_state, n_panel=3, model="test_model")
            assert result is None
