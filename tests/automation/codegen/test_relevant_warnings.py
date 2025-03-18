from collections import defaultdict
from typing import Generic, TypeVar
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from johen import generate
from pydantic import BaseModel

from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.automation.agent.models import LlmGenerateStructuredResponse
from seer.automation.codebase.models import StaticAnalysisWarning
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
    FilterWarningsOutput,
    FilterWarningsRequest,
    PrFile,
    RelevantWarningResult,
)
from seer.automation.codegen.prompts import IsFixableIssuePrompts, ReleventWarningsPrompts
from seer.automation.codegen.relevant_warnings_component import (
    AreIssuesFixableComponent,
    AssociateWarningsWithIssuesComponent,
    FetchIssuesComponent,
    FilterWarningsComponent,
    PredictRelevantWarningsComponent,
)
from seer.automation.codegen.relevant_warnings_step import (
    RelevantWarningsStep,
    RelevantWarningsStepRequest,
)
from seer.automation.models import IssueDetails, RepoDefinition, SentryEventData


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
):
    mock_repo_client = MagicMock()
    mock_pr = MagicMock()
    mock_pr_files = next(generate(list[PrFile]))
    mock_context = MagicMock()
    mock_context.get_repo_client.return_value = mock_repo_client
    mock_repo_client.repo.get_pull.return_value = mock_pr
    mock_pr.get_files.return_value = mock_pr_files

    num_associations = 5

    mock_invoke_filter_warnings_component.return_value = FilterWarningsOutput(
        warnings=next(generate(list[StaticAnalysisWarning]))
    )
    mock_invoke_fetch_issues_component.return_value = next(generate(CodeFetchIssuesOutput))
    mock_invoke_associate_warnings_with_issues_component.return_value = (
        AssociateWarningsWithIssuesOutput(
            candidate_associations=[
                (next(generate(StaticAnalysisWarning)), next(generate(IssueDetails)))
                for _ in range(num_associations)
            ]
        )
    )
    mock_invoke_are_issues_fixable_component.return_value = CodeAreIssuesFixableOutput(
        are_fixable=[True, False, None] + [True] * (num_associations - 3)
    )
    mock_invoke_predict_relevant_warnings_component.return_value = (
        CodePredictRelevantWarningsOutput(
            relevant_warning_results=[next(generate(RelevantWarningResult)) for _ in range(3)]
        )
    )

    request = RelevantWarningsStepRequest(
        repo=RepoDefinition(name="repo1", owner="owner1", provider="github", external_id="123123"),
        pr_id=123,
        callback_url="not-used-url",
        organization_id=1,
        warnings=next(generate(list[StaticAnalysisWarning])),
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

    mock_invoke_predict_relevant_warnings_component.assert_called_once()
    mock_invoke_predict_relevant_warnings_component.call_args[0][0].candidate_associations = [
        association
        for association, is_fixable in zip(
            mock_invoke_associate_warnings_with_issues_component.return_value.candidate_associations,
            mock_invoke_are_issues_fixable_component.return_value.are_fixable,
            strict=True,
        )
        if is_fixable
    ]
