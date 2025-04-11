import itertools
import textwrap
from collections import defaultdict
from typing import Generic, TypeVar
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from johen import generate
from pydantic import BaseModel

from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.automation.agent.models import LlmGenerateStructuredResponse
from seer.automation.codebase.models import StaticAnalysisRule, StaticAnalysisWarning
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    AssociateWarningsWithIssuesOutput,
    AssociateWarningsWithIssuesRequest,
    CodeAreIssuesFixableOutput,
    CodeAreIssuesFixableRequest,
    CodeFetchIssuesOutput,
    CodeFetchIssuesRequest,
    CodePredictRelevantWarningsOutput,
    CodePredictRelevantWarningsRequest,
    CodePredictStaticAnalysisSuggestionsOutput,
    CodePredictStaticAnalysisSuggestionsRequest,
    FilterWarningsOutput,
    FilterWarningsRequest,
    PrFile,
    RelevantWarningResult,
    StaticAnalysisSuggestion,
)
from seer.automation.codegen.prompts import IsFixableIssuePrompts, ReleventWarningsPrompts
from seer.automation.codegen.relevant_warnings_component import (
    AreIssuesFixableComponent,
    AssociateWarningsWithIssuesComponent,
    FetchIssuesComponent,
    FilterWarningsComponent,
    PredictRelevantWarningsComponent,
    StaticAnalysisSuggestionsComponent,
)
from seer.automation.codegen.relevant_warnings_step import (
    RelevantWarningsStep,
    RelevantWarningsStepRequest,
)
from seer.automation.models import IssueDetails, RepoDefinition, SentryEventData


@pytest.mark.parametrize(
    "warning, expected",
    [
        pytest.param(
            StaticAnalysisWarning(
                id=1,
                code="unused-variable",
                encoded_location="path/to/file.py:1~2",
                message="Warning message",
                rule=StaticAnalysisRule(
                    id=1,
                    tool="mypy",
                    code="unused-variable",
                    category="style",
                    is_autofixable=False,
                    is_stable=None,
                ),
                potentially_related_issue_titles=["Issue 1", "Issue 2"],
                encoded_code_snippet="    def test_format_code_snippet(self):\n        pass\n",
            ),
            textwrap.dedent(
                """\
                Warning ID: 1
                Warning message: Warning message
                ----------
                Location:
                    filename: path/to/file.py
                    start_line: 1
                    end_line: 2
                Code Snippet:
                ```python
                def test_format_code_snippet(self):
                    pass
                ```
                ----------
                Potentially related issue titles:
                * Issue 1
                * Issue 2
                ----------
                Static Analysis Rule:
                    Rule: unused-variable
                    Tool: mypy
                    Is auto-fixable: False
                    Is stable: None
                    Category: style

                """
            ),
            id="python-warning",
        ),
        pytest.param(
            StaticAnalysisWarning(
                id=2,
                code="unused-variable",
                encoded_location="path/to/file.js:1~2",
                message="Warning message",
                rule=StaticAnalysisRule(
                    id=2,
                    tool="eslint",
                    code="unused-variable",
                    category="style",
                    is_autofixable=False,
                    is_stable=False,
                ),
                encoded_code_snippet="",
            ),
            textwrap.dedent(
                """\
                Warning ID: 2
                Warning message: Warning message
                ----------
                Location:
                    filename: path/to/file.js
                    start_line: 1
                    end_line: 2
                Code Snippet:
                ```javascript

                ```
                ----------
                Potentially related issue titles:

                ----------
                Static Analysis Rule:
                    Rule: unused-variable
                    Tool: eslint
                    Is auto-fixable: False
                    Is stable: False
                    Category: style

                """
            ),
            id="javascript-warning-no-snippet",
        ),
    ],
)
def test_format_code_snippet(warning: StaticAnalysisWarning, expected: str):
    assert warning.format_warning() == expected


class TestFilterWarningsComponent:
    @pytest.fixture
    def component(self):
        return FilterWarningsComponent(context=MagicMock())

    def test_bad_encoded_locations_cause_errors(self, component: FilterWarningsComponent):
        warning = next(generate(StaticAnalysisWarning))
        warning.encoded_location = "../../getsentry/seer/../not/anymore.py:1:1"
        with pytest.raises(
            ValueError,
            match=f"Found `..` in the middle of path. Encoded location: {warning.encoded_location}",
        ):
            component._get_matching_pr_files(
                warning,
                [PrFile(filename="file1.py", patch="", status="modified", changes=1, sha="abc")],
            )

    def test_get_hunk_ranges(self, component: FilterWarningsComponent):
        pr_file = PrFile(
            filename="test.py",
            patch="""@@ -1,3 +1,4 @@
def hello():
    print("hello")
+    print("world")  # Line 3 is added
print("goodbye")

@@ -20,3 +21,4 @@
    print("end")
+    print("new end")  # Line 22 is added
    return""",
            status="modified",
            changes=15,
            sha="sha2",
        )
        assert component._get_sorted_hunk_ranges(pr_file) == [(1, 5), (21, 25)]

    def test_do_ranges_overlap(self, component: FilterWarningsComponent):
        # Test overlapping ranges
        assert component._do_ranges_overlap((1, 5), [(1, 5)])  # Exact match
        assert component._do_ranges_overlap((2, 4), [(1, 5)])  # Warning contained within hunk
        assert component._do_ranges_overlap((1, 3), [(2, 5)])  # Partial overlap at start
        assert component._do_ranges_overlap((4, 6), [(2, 5)])  # Partial overlap at end
        assert component._do_ranges_overlap((1, 6), [(2, 4)])  # Hunk contained within warning
        assert component._do_ranges_overlap((3, 6), [(1, 4), (5, 7)])  # Overlaps multiple hunks
        assert component._do_ranges_overlap((1, 1), [(1, 4), (5, 7)])  # Warning only has 1 line

        # Test non-overlapping ranges
        assert not component._do_ranges_overlap((1, 2), [(3, 4)])  # Warning before hunk
        assert not component._do_ranges_overlap((5, 6), [(2, 4)])  # Warning after hunk
        assert not component._do_ranges_overlap((1, 2), [])  # Empty hunks
        assert not component._do_ranges_overlap(
            (1, 2), [(10, 12), (20, 25)]
        )  # No overlap with any hunks

    class _TestInvokeTestCase(BaseModel):
        id: str

        pr_files: list[PrFile]
        "These files are relative to the repo root b/c they're from the GitHub API."

        encoded_locations_with_matches: list[str]
        "These locations come from overwatch. Currently, they may not contain the repo's full name."

        encoded_locations_without_matches: list[str]

    @pytest.mark.parametrize(
        "test_case",
        [
            _TestInvokeTestCase(
                id="getsentry/seer",
                pr_files=[
                    PrFile(
                        filename="src/seer/anomaly_detection/detectors/mp_boxcox_scorer.py",
                        patch="""@@ -233,1 +233,2 @@
                        def test_func():
                        +    print("hello")""",
                        status="modified",
                        changes=1,
                        sha="sha1",
                    )
                ],
                encoded_locations_with_matches=[
                    "src/seer/anomaly_detection/detectors/mp_boxcox_scorer.py:233:234",
                    "seer/anomaly_detection/detectors/mp_boxcox_scorer.py:233:234",
                    "getsentry/seer/src/seer/anomaly_detection/detectors/mp_boxcox_scorer.py:233:234",
                    "../../../getsentry/seer/src/seer/anomaly_detection/detectors/mp_boxcox_scorer.py:233",
                ],
                encoded_locations_without_matches=[
                    "../app/getsentry/seer/src/seer/anomaly_detection/detectors/mp_boxcox_scorer.py:233",  # too far up
                    "getsentry/seer/src/seer/anomaly_detection/mp_boxcox_scorer.py:1",  # missing detectors
                    "getsentry/seer/src/seer/detectors/mp_boxcox_scorer.py:1",  # missing anomaly_detection
                ],
            ),
            _TestInvokeTestCase(
                id="codecov/overwatch",
                pr_files=[
                    PrFile(
                        filename="app/tools/seer_signature/generate_signature.py",
                        patch="""@@ -20,3 +20,4 @@
                        def generate():
                            print("generating")
                            return True
                        +    print("done")""",
                        status="modified",
                        changes=1,
                        sha="sha1",
                    ),
                    PrFile(
                        filename="processor/tests/services/test_envelope.py",
                        patch="""@@ -4,2 +4,3 @@
                        def test_envelope():
                        +    assert True
                            pass""",
                        status="modified",
                        changes=1,
                        sha="sha1",
                    ),
                    PrFile(
                        filename="app/app/Livewire/Actions/Logout.php",
                        patch="""@@ -15,2 +15,3 @@
                        public function logout() {
                        +    return redirect('/');
                        }""",
                        status="modified",
                        changes=1,
                        sha="sha1",
                    ),
                ],
                encoded_locations_with_matches=[
                    "app/tools/seer_signature/generate_signature.py:20",
                    "tests/services/test_envelope.py:4",
                    "app/Livewire/Actions/Logout.php:15",
                ],
                encoded_locations_without_matches=[
                    "generate_signature.py:1",  # unknown location
                    "app/tools/generate_signature.py:1",  # missing seer_signature
                    "tests/services/test_package.py:1",  # wrong file
                    "app/app/Livewire/Actions/Logout.py:1",  # wrong extension
                ],
            ),
        ],
        ids=lambda test_case: test_case.id,
    )
    def test_invoke(self, test_case: _TestInvokeTestCase, component: FilterWarningsComponent):
        warnings = []
        for encoded_location in (
            test_case.encoded_locations_with_matches + test_case.encoded_locations_without_matches
        ):
            warning = next(generate(StaticAnalysisWarning))
            warning.encoded_location = encoded_location
            warnings.append(warning)

        request = FilterWarningsRequest(
            warnings=warnings,
            pr_files=test_case.pr_files,
            repo_full_name="getsentry/seer",
        )
        output: FilterWarningsOutput = component.invoke(request)
        output_encoded_locations = [warning.encoded_location for warning in output.warnings]
        assert output_encoded_locations == test_case.encoded_locations_with_matches
        assert output.warnings == warnings[: len(test_case.encoded_locations_with_matches)]


@patch("seer.rpc.DummyRpcClient.call")
class TestFetchIssuesComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=CodegenContext)
        mock_context.repo = MagicMock()
        mock_context.repo.provider = "github"
        mock_context.repo.provider_raw = "integrations:github"
        mock_context.repo.external_id = "123123"
        mock_context.run_id = 1
        return FetchIssuesComponent(mock_context)

    def test_bad_provider_raw(self, component: FetchIssuesComponent):
        mock_context = MagicMock(spec=CodegenContext)
        mock_context.repo = MagicMock()
        mock_context.repo.provider = "github"
        mock_context.repo.provider_raw = None
        mock_context.repo.external_id = "123123"
        component = FetchIssuesComponent(mock_context)
        with pytest.raises(TypeError):
            component.invoke(CodeFetchIssuesRequest(organization_id=1, pr_files=[]))

    def test_invoke_filters_files(
        self, mock_rpc_client_call: Mock, component: FetchIssuesComponent
    ):
        assert component.context.repo.provider_raw is not None
        pr_files = [
            PrFile(filename="fine.py", patch="patch1", status="modified", changes=100, sha="sha1"),
            PrFile(filename="big.py", patch="patch2", status="modified", changes=1_000, sha="sha2"),
            PrFile(filename="added.py", patch="patch3", status="added", changes=100, sha="sha3"),
        ]

        pr_filename_to_issues = {"fine.py": [next(generate(IssueDetails)).model_dump()]}
        mock_rpc_client_call.return_value = pr_filename_to_issues
        filename_to_issues_expected = {
            filename: [IssueDetails.model_validate(issue) for issue in issues]
            for filename, issues in pr_filename_to_issues.items()
        }

        request = CodeFetchIssuesRequest(
            organization_id=1,
            pr_files=pr_files,
        )
        output: CodeFetchIssuesOutput = component.invoke(request)
        assert output.filename_to_issues == filename_to_issues_expected
        assert mock_rpc_client_call.call_count == 1

        # Test the cache
        component.context.run_id += 1
        output: CodeFetchIssuesOutput = component.invoke(request)
        assert output.filename_to_issues == filename_to_issues_expected
        assert mock_rpc_client_call.call_count == 1
        component.context.run_id -= 1

        mock_rpc_client_call.return_value = None
        request.organization_id = 2
        output: CodeFetchIssuesOutput = component.invoke(request)
        assert output.filename_to_issues == {filename: [] for filename in pr_filename_to_issues}
        assert mock_rpc_client_call.call_count == 2
        request.organization_id = 1  # reset

        mock_rpc_client_call.return_value = {}
        request.organization_id = 3
        output: CodeFetchIssuesOutput = component.invoke(request)
        assert output.filename_to_issues == {filename: [] for filename in pr_filename_to_issues}


_T = TypeVar("_T")


class _MockId(Generic[_T]):
    def __init__(self, cls_with_id_attr: type[_T], id_attr: str = "id"):
        self.cls_with_id_attr = cls_with_id_attr
        self._id = 1
        self._id_attr = id_attr

    def __call__(self):
        """
        Creates a mock object of type `cls_with_id` with a unique `id` attribute.
        """
        obj = next(generate(self.cls_with_id_attr))
        setattr(obj, self._id_attr, self._id)
        self._id += 1
        return obj


_MockStaticAnalysisWarning = _MockId(StaticAnalysisWarning)
_MockIssueDetails = _MockId(IssueDetails)


def _mock_static_analysis_warning():
    """
    Creates a static analysis warning with a dummy `encoded_location` that matches the regex.
    """
    static_analysis_warning = _MockStaticAnalysisWarning()
    static_analysis_warning.encoded_location = f"{static_analysis_warning.encoded_location}.py:1"
    return static_analysis_warning


def _mock_issue_details():
    """
    Creates an issue which is guaranteed to have an event.
    """
    issue_details = _MockIssueDetails()
    issue_details.events = [next(generate(SentryEventData))]
    return issue_details


class TestAssociateWarningsWithIssuesComponent:
    @pytest.fixture
    def component(self):
        return AssociateWarningsWithIssuesComponent(context=MagicMock())

    # Patch instead of VCR so that embeddings don't depend on the texts inputted, which are
    # derived from johen-generated objects.
    @pytest.fixture(autouse=True)
    def patch_encode(self, monkeypatch: pytest.MonkeyPatch):
        rng = np.random.default_rng(seed=42)

        def mock_encode(self: GoogleProviderEmbeddings, texts: list[str]):
            output_dimensionality = self.output_dimensionality or 5
            embeddings_unnormalized = rng.random((len(texts), output_dimensionality))
            embeddings = embeddings_unnormalized / np.linalg.norm(
                embeddings_unnormalized, axis=1, keepdims=True
            )
            return embeddings

        monkeypatch.setattr(
            "seer.automation.codegen.relevant_warnings_component.GoogleProviderEmbeddings.encode",
            mock_encode,
        )

    def test_invoke(self, component: AssociateWarningsWithIssuesComponent):
        num_warnings = 3
        num_issues = 4
        max_num_associations = 5

        issues = [_mock_issue_details() for _ in range(num_issues)]
        filename_to_issues = {
            "fine.py": issues[:3],
            "fine2.py": issues[1:],  # duplicate issues w/ idxs 1 and 2 on purpose
        }
        warnings = [_mock_static_analysis_warning() for _ in range(num_warnings)]

        # We'll pick the top 5 associations among 3 warnings * 4 issues = 12 total associations.
        warning_issue_indices_expected = [(1, 5), (0, 1), (2, 5), (2, 0), (2, 1)]
        assert len(set(warning_issue_indices_expected)) == len(warning_issue_indices_expected)

        issues = [issue for issues in filename_to_issues.values() for issue in issues]
        candidate_associations_expected = [
            (warnings[warning_idx], issues[issue_idx])
            for warning_idx, issue_idx in warning_issue_indices_expected
        ]

        request = AssociateWarningsWithIssuesRequest(
            warnings=warnings,
            filename_to_issues=filename_to_issues,
            max_num_associations=max_num_associations,
        )
        output: AssociateWarningsWithIssuesOutput = component.invoke(request)
        assert len(output.candidate_associations) == max_num_associations

        # Test no duplicate associations
        warning_issue_idxs = [
            (warnings.index(warning), issues.index(issue))
            for warning, issue in output.candidate_associations
        ]
        assert len(set(warning_issue_idxs)) == len(warning_issue_idxs)

        assert output.candidate_associations == candidate_associations_expected

    def test_invoke_no_issues(self, component: AssociateWarningsWithIssuesComponent):
        request = AssociateWarningsWithIssuesRequest(
            warnings=[_mock_static_analysis_warning()],
            filename_to_issues={},
            max_num_associations=5,
        )
        output: AssociateWarningsWithIssuesOutput = component.invoke(request)
        assert output.candidate_associations == []

    def test_invoke_no_warnings(self, component: AssociateWarningsWithIssuesComponent):
        request = AssociateWarningsWithIssuesRequest(
            warnings=[],
            filename_to_issues={"fine.py": [_mock_issue_details()]},
            max_num_associations=5,
        )
        output: AssociateWarningsWithIssuesOutput = component.invoke(request)
        assert output.candidate_associations == []


class TestAreIssuesFixableComponent:
    @pytest.fixture
    def component(self):
        return AreIssuesFixableComponent(context=MagicMock())

    @pytest.fixture
    def patch_generate_structured(self, monkeypatch: pytest.MonkeyPatch):

        def mock_generate_structured(*args, **kwargs):
            completion = next(
                generate(LlmGenerateStructuredResponse[IsFixableIssuePrompts.IsIssueFixable])
            )
            # Not sure why isinstance(completion, IsFixableIssuePrompts.IsIssueFixable)
            # instead of:  isinstance(completion, LlmGenerateStructuredResponse)
            # Just force the attribute to be set:
            object.__setattr__(completion, "parsed", completion)
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.relevant_warnings_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke(self, component: AreIssuesFixableComponent, patch_generate_structured):
        max_num_issues_analyzed = 3
        issues_unique = [_mock_issue_details() for _ in range(4)]
        candidate_issues = [
            issues_unique[1],
            issues_unique[1],
            issues_unique[0],
            issues_unique[2],
            issues_unique[0],
            issues_unique[1],
            issues_unique[3],  # should be None b/c outside max_num_issues_analyzed
        ]
        # There can be duplicates when passing in issues from warning-issue associations.
        # Issues 1, 0, 2 will be analyzed for fixability.
        num_issues_analyzed_expected = 6

        request = CodeAreIssuesFixableRequest(
            candidate_issues=candidate_issues,
            max_num_issues_analyzed=max_num_issues_analyzed,
        )
        output: CodeAreIssuesFixableOutput = component.invoke(request)

        assert len(request.candidate_issues) == len(candidate_issues)
        assert len(output.are_fixable) == len(request.candidate_issues)

        # Test that num_issues_analyzed_expected were analyzed
        assert (
            sum(is_fixable is not None for is_fixable in output.are_fixable)
            == num_issues_analyzed_expected
        )
        assert all(is_fixable is not None for is_fixable in output.are_fixable[:-1])
        assert output.are_fixable[-1] is None

        # Test that the same issues (by id) have the same is_fixable value
        issue_id_to_are_fixable = defaultdict(set)
        for issue, is_fixable in zip(candidate_issues, output.are_fixable, strict=True):
            issue_id_to_are_fixable[issue.id].add(is_fixable)
        for issue_id, are_fixable in issue_id_to_are_fixable.items():
            assert len(are_fixable) == 1, issue_id

    @pytest.fixture
    def patch_generate_structured_failure(self, monkeypatch: pytest.MonkeyPatch):

        def mock_generate_structured(*args, **kwargs):
            completion = next(
                generate(LlmGenerateStructuredResponse[IsFixableIssuePrompts.IsIssueFixable])
            )
            object.__setattr__(completion, "parsed", None)
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.relevant_warnings_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke_with_failed_completion(
        self, component: AreIssuesFixableComponent, patch_generate_structured_failure
    ):
        max_num_issues_analyzed = 1
        candidate_issues = [_mock_issue_details()]

        request = CodeAreIssuesFixableRequest(
            candidate_issues=candidate_issues,
            max_num_issues_analyzed=max_num_issues_analyzed,
        )
        output: CodeAreIssuesFixableOutput = component.invoke(request)

        assert len(request.candidate_issues) == len(candidate_issues)
        assert output.are_fixable == [True]  # b/c it defaults to fixable


class TestPredictRelevantWarningsComponent:
    @pytest.fixture
    def component(self):
        return PredictRelevantWarningsComponent(context=MagicMock())

    @pytest.fixture(autouse=True)
    def patch_generate_structured(self, monkeypatch: pytest.MonkeyPatch):

        def mock_generate_structured(*args, **kwargs):
            completion = next(
                generate(
                    LlmGenerateStructuredResponse[ReleventWarningsPrompts.DoesFixingWarningFixIssue]
                )
            )
            object.__setattr__(completion, "parsed", completion)
            return completion

        monkeypatch.setattr(
            "seer.automation.codegen.relevant_warnings_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke(self, component: PredictRelevantWarningsComponent):
        candidate_associations = [
            (_mock_static_analysis_warning(), _mock_issue_details()) for _ in range(4)
        ]
        request = CodePredictRelevantWarningsRequest(candidate_associations=candidate_associations)
        output: CodePredictRelevantWarningsOutput = component.invoke(request)

        for (warning, issue), result in zip(
            candidate_associations, output.relevant_warning_results, strict=True
        ):
            assert warning.id == result.warning_id
            assert issue.id == result.issue_id


@patch(
    "seer.automation.codegen.relevant_warnings_component.StaticAnalysisSuggestionsComponent.invoke"
)
@patch("seer.automation.codegen.relevant_warnings_component.FilterWarningsComponent.invoke")
@patch("seer.automation.codegen.relevant_warnings_component.FetchIssuesComponent.invoke")
@patch("seer.automation.codegen.relevant_warnings_component.AreIssuesFixableComponent.invoke")
@patch(
    "seer.automation.codegen.relevant_warnings_component.AssociateWarningsWithIssuesComponent.invoke"
)
@patch(
    "seer.automation.codegen.relevant_warnings_component.PredictRelevantWarningsComponent.invoke"
)
@patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
@patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
def test_relevant_warnings_step_invoke(
    mock_instantiate_context: Mock,
    mock_pipeline_step: MagicMock,
    mock_invoke_predict_relevant_warnings_component: Mock,
    mock_invoke_associate_warnings_with_issues_component: Mock,
    mock_invoke_are_issues_fixable_component: Mock,
    mock_invoke_fetch_issues_component: Mock,
    mock_invoke_filter_warnings_component: Mock,
    mock_invoke_static_analysis_suggestions_component: Mock,
):
    mock_repo_client = MagicMock()
    mock_pr = MagicMock()
    mock_pr_files = next(generate(list[PrFile]))
    mock_context = MagicMock()
    mock_context.get_repo_client.return_value = mock_repo_client
    mock_repo_client.repo.get_pull.return_value = mock_pr
    mock_pr.get_files.return_value = mock_pr_files

    mock_all_warnings = next(generate(list[StaticAnalysisWarning]))
    mock_invoke_filter_warnings_component.return_value = FilterWarningsOutput(
        warnings=mock_all_warnings
    )
    mock_issues_fetched = next(generate(CodeFetchIssuesOutput))
    mock_invoke_fetch_issues_component.return_value = mock_issues_fetched
    all_selected_issues = list(
        itertools.chain.from_iterable(mock_issues_fetched.filename_to_issues.values())
    )
    all_selected_issues_count = len(all_selected_issues)
    assert all_selected_issues_count > 0, "No issues fetched"

    mock_invoke_associate_warnings_with_issues_component.return_value = (
        AssociateWarningsWithIssuesOutput(
            candidate_associations=[
                (warning, issue)
                for widx, warning in enumerate(mock_all_warnings)
                for iidx, issue in enumerate(all_selected_issues)
                if widx % 2 == 0 and iidx % 3 == 0
            ]
        )
    )
    mock_invoke_are_issues_fixable_component.return_value = CodeAreIssuesFixableOutput(
        are_fixable=[False] + [True] * (all_selected_issues_count - 1)
    )
    mock_invoke_static_analysis_suggestions_component.return_value = (
        CodePredictStaticAnalysisSuggestionsOutput(
            suggestions=[next(generate(StaticAnalysisSuggestion)) for _ in range(3)]
        )
    )

    request = RelevantWarningsStepRequest(
        repo=RepoDefinition(name="repo1", owner="owner1", provider="github", external_id="123123"),
        pr_id=123,
        callback_url="not-used-url",
        organization_id=1,
        warnings=mock_all_warnings,
        commit_sha="sha123",
        run_id=1,
        max_num_associations=10,
        max_num_issues_analyzed=10,
        should_post_to_overwatch=False,
    )
    step = RelevantWarningsStep(request=request)
    step.context = mock_context
    step.invoke()

    mock_context.get_repo_client.assert_called_once()
    mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)

    mock_invoke_filter_warnings_component.assert_called_once()
    mock_invoke_filter_warnings_component.call_args[0][0].warnings = request.warnings
    mock_invoke_filter_warnings_component.call_args[0][0].pr_files = mock_pr_files

    mock_invoke_fetch_issues_component.assert_called_once()
    mock_invoke_fetch_issues_component.call_args[0][0].organization_id = request.organization_id
    mock_invoke_fetch_issues_component.call_args[0][0].pr_files = mock_pr_files

    mock_invoke_associate_warnings_with_issues_component.assert_called_once()
    mock_invoke_associate_warnings_with_issues_component.call_args[0][0].warnings = request.warnings
    mock_invoke_associate_warnings_with_issues_component.call_args[0][
        0
    ].filename_to_issues = mock_invoke_fetch_issues_component.return_value.filename_to_issues
    mock_invoke_associate_warnings_with_issues_component.call_args[0][
        0
    ].max_num_associations = request.max_num_associations

    mock_invoke_are_issues_fixable_component.assert_called_once()
    mock_invoke_are_issues_fixable_component.call_args[0][0].candidate_issues = [
        issue
        for _, issue in mock_invoke_associate_warnings_with_issues_component.return_value.candidate_associations
    ]
    mock_invoke_are_issues_fixable_component.call_args[0][
        0
    ].max_num_issues_analyzed = request.max_num_issues_analyzed

    mock_invoke_static_analysis_suggestions_component.assert_called_once()
    mock_invoke_static_analysis_suggestions_component.call_args[0][0].pr_files = mock_pr_files
    mock_invoke_static_analysis_suggestions_component.call_args[0][0].warnings = mock_all_warnings
    mock_invoke_static_analysis_suggestions_component.call_args[0][0].fixable_issues = (
        all_selected_issues[1:]
    )


class TestStaticAnalysisSuggestionsComponent:
    @pytest.fixture
    def component(self):
        return StaticAnalysisSuggestionsComponent(context=MagicMock())

    @pytest.fixture(autouse=True)
    def patch_generate_structured(self, monkeypatch: pytest.MonkeyPatch):
        def mock_generate_structured(*args, **kwargs):
            # Create a mock response object
            mock_response = MagicMock()
            # Create a list of suggestions
            suggestions = [
                StaticAnalysisSuggestion(
                    path="test/path/file.py",
                    line=42,
                    short_description="Test suggestion",
                    justification="Test justification",
                    related_warning_id="123",
                    related_issue_id="456",
                    severity_score=0.8,
                    confidence_score=0.9,
                    missing_evidence=["More context", "Test cases"],
                )
            ]
            # Set the parsed attribute directly
            mock_response.parsed = suggestions
            return mock_response

        monkeypatch.setattr(
            "seer.automation.codegen.relevant_warnings_component.LlmClient.generate_structured",
            mock_generate_structured,
        )

    def test_invoke(self, component: StaticAnalysisSuggestionsComponent):
        # Create test data
        pr_files = [
            PrFile(
                filename="test/path/file.py",
                patch="@@ -1,3 +1,4 @@\ndef hello():\n    print('hello')\n+    print('world')",
                status="modified",
                changes=1,
                sha="sha1",
            )
        ]
        warnings = [_mock_static_analysis_warning() for _ in range(2)]
        fixable_issues = [_mock_issue_details() for _ in range(2)]

        request = CodePredictStaticAnalysisSuggestionsRequest(
            pr_files=pr_files,
            warnings=warnings,
            fixable_issues=fixable_issues,
        )

        output = component.invoke(request)
        assert output is not None
        assert len(output.suggestions) == 1
        assert output.suggestions[0].path == "test/path/file.py"
        assert output.suggestions[0].line == 42
        assert output.suggestions[0].short_description == "Test suggestion"
        assert output.suggestions[0].justification == "Test justification"
        assert output.suggestions[0].related_warning_id == "123"
        assert output.suggestions[0].related_issue_id == "456"
        assert output.suggestions[0].severity_score == 0.8
        assert output.suggestions[0].confidence_score == 0.9
        assert output.suggestions[0].missing_evidence == ["More context", "Test cases"]

    def test_invoke_no_suggestions(
        self, component: StaticAnalysisSuggestionsComponent, monkeypatch
    ):
        # Override the mock to return None for parsed
        def mock_generate_structured_none(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.parsed = None
            return mock_response

        monkeypatch.setattr(
            "seer.automation.codegen.relevant_warnings_component.LlmClient.generate_structured",
            mock_generate_structured_none,
        )

        # Create test data
        pr_files = [
            PrFile(
                filename="test/path/file.py",
                patch="@@ -1,3 +1,4 @@\ndef hello():\n    print('hello')\n+    print('world')",
                status="modified",
                changes=1,
                sha="sha1",
            )
        ]
        warnings = [_mock_static_analysis_warning() for _ in range(2)]
        fixable_issues = [_mock_issue_details() for _ in range(2)]

        request = CodePredictStaticAnalysisSuggestionsRequest(
            pr_files=pr_files,
            warnings=warnings,
            fixable_issues=fixable_issues,
        )

        output = component.invoke(request)
        assert output is None

    def test_format_issue(self, component: StaticAnalysisSuggestionsComponent):
        # Create a test issue
        issue = _mock_issue_details()

        # Call the private method using a workaround
        formatted_issue = component._format_issue(issue)

        # Verify the format
        assert f"<sentry_issue><issue_id>{issue.id}</issue_id>" in formatted_issue
        assert "<title>" in formatted_issue
        assert "</sentry_issue>" in formatted_issue


# Tests for to_overwatch_format method
@pytest.mark.parametrize(
    "suggestion,expected",
    [
        (
            StaticAnalysisSuggestion(
                path="src/main.py",
                line=42,
                short_description="Test warning",
                justification="This is a test justification",
                related_warning_id="123",
                related_issue_id="456",
                severity_score=0.8,
                confidence_score=0.9,
                missing_evidence=[],
            ),
            {
                "warning_id": 123,
                "issue_id": 456,
                "does_fixing_warning_fix_issue": True,
                "relevance_probability": 0.8 * 0.9,
                "reasoning": "This is a test justification",
                "short_justification": "This is a test justification",
                "short_description": "Test warning",
                "encoded_location": "src/main.py:42",
            },
        ),
        (
            StaticAnalysisSuggestion(
                path="tests/test_file.py",
                line=100,
                short_description="Another warning",
                justification="Another justification",
                related_warning_id="789",
                related_issue_id="101",
                severity_score=0.5,
                confidence_score=0.6,
                missing_evidence=["evidence1", "evidence2"],
            ),
            {
                "warning_id": 789,
                "issue_id": 101,
                "does_fixing_warning_fix_issue": True,
                "relevance_probability": 0.5 * 0.6,
                "reasoning": "Another justification",
                "short_justification": "Another justification",
                "short_description": "Another warning",
                "encoded_location": "tests/test_file.py:100",
            },
        ),
        (
            StaticAnalysisSuggestion(
                path="complex/path/to/file.py",
                line=1,
                short_description="Minimal warning",
                justification="Minimal justification",
                related_warning_id=None,
                related_issue_id=None,
                severity_score=0.1,
                confidence_score=0.2,
                missing_evidence=[],
            ),
            {
                "warning_id": None,
                "issue_id": None,
                "does_fixing_warning_fix_issue": True,
                "relevance_probability": 0.1 * 0.2,
                "reasoning": "Minimal justification",
                "short_justification": "Minimal justification",
                "short_description": "Minimal warning",
                "encoded_location": "complex/path/to/file.py:1",
            },
        ),
    ],
)
def test_to_overwatch_format(suggestion, expected):
    """Test the to_overwatch_format method with various inputs."""
    result = suggestion.to_overwatch_format()

    # Verify the result is a RelevantWarningResult
    assert isinstance(result, RelevantWarningResult)

    # Verify all fields match expected values
    for key, value in expected.items():
        assert (
            getattr(result, key) == value
        ), f"Mismatch in {key}: expected {value}, got {getattr(result, key)}"
