import textwrap
from unittest.mock import MagicMock, Mock, patch

import pytest

from seer.automation.agent.models import LlmGenerateStructuredResponse
from seer.automation.codebase.models import PrFile
from seer.automation.codegen.bug_prediction_component import (
    BugPredictorComponent,
    FilterFilesComponent,
    FormatterComponent,
)
from seer.automation.codegen.bug_prediction_step import BugPredictionStep
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    BugPrediction,
    BugPredictorHypothesis,
    BugPredictorOutput,
    BugPredictorRequest,
    FilterFilesOutput,
    FilterFilesRequest,
    FormatterOutput,
    FormatterRequest,
)
from seer.automation.models import RepoDefinition


@pytest.fixture
def mock_pr_files():
    return [
        PrFile(
            filename="src/main.py",
            patch=textwrap.dedent(
                """\
                @@ -1,2 +1,2 @@
                    def main():
                -       return 1
                +       return 2
                """
            ),
            status="modified",
            changes=1,
            sha="sha1",
            previous_filename="src/main.py",
            repo_full_name="getsentry/seer",
        ),
        PrFile(
            filename="src/utils.py",
            patch=textwrap.dedent(
                """\
                @@ -1,2 +1,2 @@
                    def helper():
                -       x = 1
                +       x = 2
                """
            ),
            status="modified",
            changes=1,
            sha="sha2",
            previous_filename="src/utils.py",
            repo_full_name="getsentry/seer",
        ),
        PrFile(
            filename="src/renamed_file.py",
            patch="",
            status="renamed",
            changes=0,
            sha="sha3",
            previous_filename="src/old_name.py",
            repo_full_name="getsentry/seer",
        ),
        PrFile(
            filename="src/removed_file.py",
            patch=textwrap.dedent(
                """\
                @@ -1,5 +0,0 @@
                -def to_be_removed():
                -    '''
                -    This function will be removed
                -    '''
                -    return None
                """
            ),
            status="removed",
            changes=5,
            sha="sha4",
            previous_filename="src/removed_file.py",
            repo_full_name="getsentry/seer",
        ),
    ]


@pytest.fixture
def mock_bug_predictions():
    return [
        BugPrediction(
            title="IndexError in root_cause_step.py can crash the server",
            short_description="The code in `root_cause_step.py` might crash the server due to an `IndexError` when accessing `root_cause_output.causes[0].id` without checking if `root_cause_output.causes` is empty.",
            description="In `root_cause_step.py`, the code directly accesses `root_cause_output.causes[0].id` without verifying that `root_cause_output.causes` is not empty. If the root cause analysis returns an empty list of causes, accessing the first element will raise an `IndexError`, which is not caught and can crash the server. This scenario occurs when the automated root cause analysis fails to identify any potential causes. The `event_manager.send_root_cause_analysis_result` method and other parts of the codebase already handle the empty causes case, indicating that this condition is expected and should be handled gracefully. Failing to do so in `root_cause_step.py` introduces a critical vulnerability.",
            suggested_fix="Add a check to ensure `root_cause_output.causes` is not empty before accessing `root_cause_output.causes[0].id`. If it's empty, handle the case gracefully, possibly by skipping the subsequent steps or logging an error.",
            encoded_location="root_cause_step.py",
            severity=0.9,
            confidence=0.9,
        ),
        BugPrediction(
            title="Potential null reference in `resolve_comment_thread` could crash server",
            short_description="The code might crash due to accessing `.id` on a potentially null `active_comment_thread` in `resolve_comment_thread`. This could happen if a thread is resolved multiple times.",
            description="The `resolve_comment_thread` function attempts to access `cur.steps[step_index].active_comment_thread.id` without checking if `active_comment_thread` is None. If `active_comment_thread` is None, this will raise an AttributeError, crashing the request. This can occur if a thread is resolved multiple times, or if there's a race condition. For example, if `active_comment_thread` is already None due to a previous resolve operation, accessing `.id` will cause a crash.",
            suggested_fix="Add a null check before accessing the `id` property of `active_comment_thread`. Change `elif request.payload.thread_id == cur.steps[step_index].active_comment_thread.id:` to `elif cur.steps[step_index].active_comment_thread and request.payload.thread_id == cur.steps[step_index].active_comment_thread.id:`",
            encoded_location="src/seer/automation/autofix/tasks.py",
            severity=0.9,
            confidence=0.9,
        ),
    ]


class TestFilterFilesComponent:
    @pytest.fixture
    def component(self):
        return FilterFilesComponent(context=MagicMock())

    @pytest.fixture
    def patch_generate_structured(self, monkeypatch: pytest.MonkeyPatch):
        def mock_generate_structured(*args, **kwargs):
            completion = MagicMock(spec=LlmGenerateStructuredResponse)
            # Return the first file as the filtered result
            completion.parsed = ["src/main.py"]
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.bug_prediction_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke_no_filtering_needed(self, component: FilterFilesComponent, mock_pr_files):
        request = FilterFilesRequest(
            pr_files=mock_pr_files[:1], pr_title="Title", pr_body="Body", num_files_desired=2
        )
        output = component.invoke(request)
        assert output.pr_files == [mock_pr_files[0]]

    def test_invoke_with_filtering(
        self, component: FilterFilesComponent, mock_pr_files, patch_generate_structured
    ):
        request = FilterFilesRequest(
            pr_files=mock_pr_files, pr_title="Title", pr_body="Body", num_files_desired=1
        )
        output = component.invoke(request)
        assert output.pr_files == [mock_pr_files[pr_idx] for pr_idx in [0, 2, 3]]
        # 0 b/c of mock. 2 and 3 b/c their hunks won't be shown

    @pytest.fixture
    def patch_generate_structured_failure(self, monkeypatch: pytest.MonkeyPatch):
        def mock_generate_structured(*args, **kwargs):
            completion = MagicMock(spec=LlmGenerateStructuredResponse)
            completion.parsed = None
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.bug_prediction_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke_with_filtering_failure(
        self,
        component: FilterFilesComponent,
        mock_pr_files: list[PrFile],
        patch_generate_structured_failure,
    ):
        request = FilterFilesRequest(
            pr_files=mock_pr_files,
            pr_title="Title",
            pr_body="Body",
            num_files_desired=len(mock_pr_files),
            shuffle_files=False,
        )
        output = component.invoke(request)

        assert isinstance(output, FilterFilesOutput)
        assert len(output.pr_files) == request.num_files_desired
        assert output.pr_files == mock_pr_files


class TestFormatterComponent:
    @pytest.fixture
    def component(self):
        return FormatterComponent(context=MagicMock())

    @pytest.fixture
    def patch_generate_structured(
        self, monkeypatch: pytest.MonkeyPatch, mock_bug_predictions: list[BugPrediction]
    ):
        def mock_generate_structured(*args, **kwargs):
            completion = MagicMock(spec=LlmGenerateStructuredResponse)
            completion.parsed = mock_bug_predictions
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.bug_prediction_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke_with_valid_prediction(
        self,
        component: FormatterComponent,
        mock_bug_predictions,
        patch_generate_structured,
    ):
        followups = [
            "Some ~200-500 word followups from the agent",
            "Another ~200-500 word followup from the agent",
            "A third ~200-500 word followup from the agent",
        ]

        request = FormatterRequest(followups=followups)
        result = component.invoke(request)

        assert isinstance(result, FormatterOutput)
        assert result.bug_predictions == mock_bug_predictions

    def test_invoke_with_empty_followups(self, component):
        request = FormatterRequest(followups=[])
        result = component.invoke(request)

        assert isinstance(result, FormatterOutput)
        assert len(result.bug_predictions) == 0

    def test_invoke_with_none_followups(self, component):
        request = FormatterRequest(followups=[None, ""])
        result = component.invoke(request)

        assert isinstance(result, FormatterOutput)
        assert len(result.bug_predictions) == 0

    @pytest.fixture
    def patch_generate_structured_failure(self, monkeypatch: pytest.MonkeyPatch):
        def mock_generate_structured(*args, **kwargs):
            completion = MagicMock(spec=LlmGenerateStructuredResponse)
            completion.parsed = None
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.bug_prediction_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke_with_parsing_failure(self, component, patch_generate_structured_failure):
        request = FormatterRequest(followups=["Test prediction"])
        result = component.invoke(request)

        assert isinstance(result, FormatterOutput)
        assert len(result.bug_predictions) == 0


class TestBugPredictorComponent:
    @pytest.fixture
    def mock_repo_def(self):
        return RepoDefinition(name="repo1", owner="owner1", provider="github", external_id="123123")

    @pytest.fixture
    def component(self, mock_repo_def: RepoDefinition):
        mock_context = MagicMock(spec=CodegenContext)
        mock_context.repo = mock_repo_def
        mock_context.get_repo_client.return_value = MagicMock()
        mock_context.state = MagicMock()
        return BugPredictorComponent(context=mock_context)

    @pytest.fixture
    def mock_hypotheses(self):
        return [
            BugPredictorHypothesis(
                content="Hypothesis 1",
                location_filename="src/main.py",
                location_start_line_num=1,
                location_end_line_num=2,
            ),
            BugPredictorHypothesis(
                content="Hypothesis 2",
                location_filename="src/utils.py",
                location_start_line_num=3,
                location_end_line_num=4,
            ),
        ]

    @pytest.fixture
    def mock_hypotheses_unstructured(self):
        return "Some bug hypotheses, locations, and further questions to answer."

    @pytest.fixture
    def mock_followup_prefix(self):
        return "Some analysis for hypothesis "

    @pytest.fixture
    def patch_llm_agent(
        self,
        mock_hypotheses_unstructured: str,
        mock_followup_prefix: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def mock_agent_run(*args, **kwargs):
            run_name = kwargs["run_config"].run_name
            if run_name == "Draft bug hypotheses":
                return mock_hypotheses_unstructured
            elif run_name.startswith("Follow up hypothesis "):
                hypothesis_num = int(run_name.split(" ")[-1])
                return f"{mock_followup_prefix}{hypothesis_num}"
            else:
                raise ValueError(f"Unexpected run name: {run_name}")

        monkeypatch.setattr(
            "seer.automation.codegen.bug_prediction_component.LlmAgent.run",
            mock_agent_run,
        )

    @pytest.fixture
    def patch_generate_structured(self, monkeypatch: pytest.MonkeyPatch, mock_hypotheses):
        def mock_generate_structured(*args, **kwargs):
            completion = MagicMock(spec=LlmGenerateStructuredResponse)
            completion.parsed = mock_hypotheses
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.bug_prediction_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke(
        self,
        component: BugPredictorComponent,
        mock_pr_files: list[PrFile],
        mock_repo_def: RepoDefinition,
        mock_hypotheses_unstructured: str,
        mock_hypotheses: list[BugPredictorHypothesis],
        mock_followup_prefix: str,
        patch_llm_agent,
        patch_generate_structured,
        monkeypatch: pytest.MonkeyPatch,
    ):
        bug_predictor_request = BugPredictorRequest(
            pr_files=mock_pr_files,
            repo_full_name=mock_repo_def.full_name,
            pr_title="Test PR",
            pr_body="Test description",
        )
        bug_predictor_output = component.invoke(bug_predictor_request)

        assert isinstance(bug_predictor_output, BugPredictorOutput)
        assert bug_predictor_output.hypotheses_unstructured == mock_hypotheses_unstructured
        assert bug_predictor_output.hypotheses == mock_hypotheses
        assert bug_predictor_output.followups == [
            f"{mock_followup_prefix}{hypothesis_num}"
            for hypothesis_num in range(1, len(bug_predictor_output.hypotheses) + 1)
        ]


@patch("seer.automation.codegen.bug_prediction_component.FormatterComponent.invoke")
@patch("seer.automation.codegen.bug_prediction_component.BugPredictorComponent.invoke")
@patch("seer.automation.codegen.bug_prediction_component.FilterFilesComponent.invoke")
@patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
@patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
@patch("seer.automation.codegen.bug_prediction_step.BugPredictionStep._post_results_to_overwatch")
def test_bug_prediction_step_invoke(
    mock_post_results_to_overwatch: Mock,
    mock_pipeline_step: MagicMock,
    mock_instantiate_context: Mock,
    mock_invoke_filter_files_component: Mock,
    mock_invoke_bug_predictor_component: Mock,
    mock_invoke_formatter_component: Mock,
):
    mock_repo_client = MagicMock()
    mock_pr = MagicMock()
    mock_pr_files = [
        PrFile(
            filename="src/main.py",
            patch="@@ -1,2 +1,2 @@\n def test():\n-    return 1\n+    return 2",
            status="modified",
            changes=1,
            sha="abc123",
            previous_filename="src/main.py",
            repo_full_name="owner1/repo1",
        )
    ]
    mock_filtered_pr_files = mock_pr_files[:1]

    mock_context = MagicMock()
    mock_context.get_repo_client.return_value = mock_repo_client
    mock_repo_client.repo.get_pull.return_value = mock_pr
    mock_pr.get_files.return_value = mock_pr_files
    mock_pr.title = "Test PR"
    mock_pr.body = "Test PR description"

    mock_invoke_filter_files_component.return_value = FilterFilesOutput(
        pr_files=mock_filtered_pr_files
    )

    bug_predictor_followups = ["Potential bug: Off-by-one error in array access"]
    mock_invoke_bug_predictor_component.return_value = BugPredictorOutput(
        hypotheses_unstructured="Hypothesis text",
        hypotheses=[
            BugPredictorHypothesis(
                content="Hypothesis content",
                location_filename="src/main.py",
                location_start_line_num=1,
                location_end_line_num=2,
            )
        ],
        followups=bug_predictor_followups,
    )

    bug_predictions = [
        BugPrediction(
            title="Bug prediction 1",
            description="Bug prediction 1 description",
            short_description="Bug prediction 1 short description",
            suggested_fix="Bug prediction 1 suggested fix",
            encoded_location="Bug prediction 1 encoded location",
            severity=0.5,
            confidence=0.5,
        )
    ]
    mock_invoke_formatter_component.return_value = FormatterOutput(bug_predictions=bug_predictions)

    request = {
        "repo": {
            "name": "repo1",
            "owner": "owner1",
            "provider": "github",
            "external_id": "123123",
        },
        "pr_id": 123,
        "callback_url": "not-used-url",
        "organization_id": 1,
        "warnings": [],
        "commit_sha": "sha123",
        "run_id": 1,
        "should_post_to_overwatch": False,
    }
    step = BugPredictionStep(request)
    step.context = mock_context
    step.invoke()

    mock_context.get_repo_client.assert_called_once()
    mock_repo_client.repo.get_pull.assert_called_once_with(request["pr_id"])

    mock_invoke_filter_files_component.assert_called_once()
    assert mock_invoke_filter_files_component.call_args[0][0].pr_files == mock_pr_files

    mock_invoke_bug_predictor_component.assert_called_once()
    bug_predictor_request = mock_invoke_bug_predictor_component.call_args[0][0]
    assert bug_predictor_request.pr_files == mock_filtered_pr_files
    assert bug_predictor_request.pr_title == "Test PR"
    assert bug_predictor_request.pr_body == "Test PR description"

    mock_invoke_formatter_component.assert_called_once()
    formatter_input = mock_invoke_formatter_component.call_args[0][0]
    assert formatter_input.followups == bug_predictor_followups

    mock_post_results_to_overwatch.assert_called_once()
    assert mock_post_results_to_overwatch.call_args[0][0].bug_predictions == bug_predictions
