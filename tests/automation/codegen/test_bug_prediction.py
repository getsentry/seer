import textwrap
from unittest.mock import MagicMock, Mock, patch

import pytest

from seer.automation.agent.models import LlmGenerateStructuredResponse
from seer.automation.codebase.models import PrFile
from seer.automation.codegen.bug_prediction_component import (
    BugPredictorComponent,
    FilterFilesComponent,
)
from seer.automation.codegen.bug_prediction_step import BugPredictionStep
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    BugPredictorHypothesis,
    BugPredictorOutput,
    BugPredictorRequest,
    FilterFilesOutput,
    FilterFilesRequest,
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

    mock_post_results_to_overwatch.assert_called_once()
