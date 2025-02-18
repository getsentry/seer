from unittest.mock import MagicMock, patch

from johen import generate

from seer.automation.codebase.models import StaticAnalysisWarning
from seer.automation.codegen.models import (
    AssociateWarningsWithIssuesOutput,
    CodeAreIssuesFixableOutput,
    CodeFetchIssuesOutput,
    CodePredictRelevantWarningsOutput,
    PrFile,
)
from seer.automation.codegen.relevant_warnings_step import (
    RelevantWarningsStep,
    RelevantWarningsStepRequest,
)

# from seer.automation.codegen.relevant_warnings_component import (
#     AreIssuesFixableComponent,
#     AssociateWarningsWithIssuesComponent,
#     FetchIssuesComponent,
#     PredictRelevantWarningsComponent,
# )
from seer.automation.models import IssueDetails, RelevantWarningResult, RepoDefinition

# @pytest.fixture
# def context():
#     request = next(generate(CodegenRelevantWarningsRequest))
#     continuation = CodegenContinuation(request=request)
#     state = CodegenContinuationState(val=continuation)
#     event_manager = CodegenEventManager(state=state)
#     return CodegenContext(state=state, event_manager=event_manager, repo=request.repo)

# TODO: test the 4 components individually, prolly VCR them.


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
    mock_instantiate_context,
    mock_pipeline_step,
    mock_invoke_predict_relevant_warnings_component,
    mock_invoke_associate_warnings_with_issues_component,
    mock_invoke_are_issues_fixable_component,
    mock_invoke_fetch_issues_component,
):
    mock_repo_client = MagicMock()
    mock_pr = MagicMock()
    mock_pr_files = next(generate(list[PrFile]))
    mock_context = MagicMock()
    mock_context.get_repo_client.return_value = mock_repo_client
    mock_repo_client.repo.get_pull.return_value = mock_pr
    mock_pr.get_files.return_value = mock_pr_files

    num_associations = 5

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
        organization_id=1,
        warnings=next(generate(list[StaticAnalysisWarning])),
        run_id=1,
        max_num_associations=10,
        max_num_issues_analyzed=10,
    )
    step = RelevantWarningsStep(request=request)
    step.context = mock_context
    step.invoke()

    mock_context.get_repo_client.assert_called_once()
    mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
    mock_pr.get_files.assert_called_once()

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
