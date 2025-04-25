import logging
from unittest import mock

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe

from seer.automation.codebase.models import PrFile
from seer.automation.codegen.models import (
    CodeFetchIssuesOutput,
    CodegenRelevantWarningsRequest,
    StaticAnalysisSuggestion,
)
from seer.automation.codegen.relevant_warnings_step import RelevantWarningsStep

logger = logging.getLogger(__name__)


@observe(name="[Relevant Warnings Eval] Sync run evaluation on item")
def sync_run_evaluation_on_item(
    item: DatasetItemClient,
) -> list[StaticAnalysisSuggestion] | None:
    """
    Actually executes the RelevantWarningsStep with input from the EvalItem.
    It mocks the following parts of the pipeline:
      - Fetching the PR diff
      - Fetching the Sentry issues
    """

    # Build the input data for the evaluation. This is part of the EvalItem.
    request = CodegenRelevantWarningsRequest.model_validate(item.input["request"])
    # Make sure we don't post to Overwatch.
    request.should_post_to_overwatch = False
    pr_files = [PrFile.model_validate(file) for file in item.input["pr_files"]]
    raw_issues = item.input["issues"]

    relevant_warnings_step = RelevantWarningsStep(
        request=request.model_dump(),
    )
    # Mock the repo client to return the pr_files
    mock_repo_client = mock.Mock()
    mock_repo_client.repo.get_pull.return_value.get_files.return_value = pr_files
    relevant_warnings_step.context.get_repo_client = mock.Mock(return_value=mock_repo_client)
    # Mock FilterIssuesComponent to return the issues
    # Ignoring the filename_to_issues part of the output.
    mock_filter_issues = mock.patch(
        "seer.automation.codegen.relevant_warnings_step.FilterIssuesComponent.invoke",
        return_value=CodeFetchIssuesOutput(filename_to_issues={"all_issues": raw_issues}),
    )
    relevant_warnings_step.context.filter_issues_component = mock_filter_issues.start()

    # Invoke the step.
    relevant_warnings_step.invoke()

    # Grab the suggestions from the state.
    state = relevant_warnings_step.context.state
    return state.static_analysis_suggestions


def score_suggestions_length(
    suggestions: list[StaticAnalysisSuggestion],
    item: DatasetItemClient,
) -> float:
    """
    Scores that the number of suggestions we got is close to the number of expected suggestions.

    """
    expected_suggestions = item.expected_output["suggestions"]
    return abs(len(suggestions) - len(expected_suggestions))


def score_suggestions_content(
    suggestion: StaticAnalysisSuggestion,
    item: DatasetItemClient,
) -> float:
    # expected_suggestions = item.expected_output["suggestions"]
    # TODO: Implement scoring logic.
    # This will totally require a LLM call.
    return 0.0
