import logging
from unittest import mock

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.codegen.evals.models import (
    EvalItemInput,
    EvalItemOutput,
    ModelEvaluationOutput,
    ModelEvaluationOutputList,
)
from seer.automation.codegen.models import CodeFetchIssuesOutput, StaticAnalysisSuggestion
from seer.automation.codegen.relevant_warnings_step import RelevantWarningsStep
from seer.automation.codegen.tasks import create_initial_relevant_warnings_run
from seer.automation.state import DbStateRunTypes

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

    item = EvalItemInput.model_validate(item)

    # Build the request from the item.
    request = item.get_request()
    # Make sure we don't post to Overwatch.
    request.should_post_to_overwatch = False

    # Create a proper state for the evaluation run
    state = create_initial_relevant_warnings_run(request)
    run_id = state.get().run_id

    relevant_warnings_step = RelevantWarningsStep(
        request={"run_id": run_id, **request.model_dump()},
        type=DbStateRunTypes.RELEVANT_WARNINGS,
    )

    # Mock parts of the pipeline depending on the extra info saved in the EvalItem.
    # If it's not mocked we will reach out to GitHub at evaluation time.
    if item.pr_files:
        # Mock the repo client to return the pr_files
        mock_repo_client = mock.Mock()
        mock_repo_client.repo.get_pull.return_value.get_files.return_value = item.pr_files
        relevant_warnings_step.context.get_repo_client = mock.Mock(return_value=mock_repo_client)  # type: ignore [method-assign]

    # Mock FilterIssuesComponent to return the issues
    # Ignoring the filename_to_issues part of the output.
    mock.patch(
        "seer.automation.codegen.relevant_warnings_step.FetchIssuesComponent.invoke",
        return_value=CodeFetchIssuesOutput(filename_to_issues={"all_issues": item.issues}),
    )

    # Invoke the step.
    relevant_warnings_step.invoke()

    # Grab the suggestions from the state.
    state = relevant_warnings_step.context.state
    return state.get().static_analysis_suggestions


@observe(name="[Relevant Warnings Eval] Evaluate semantic similarity")
def evaluate_if_bug_is_in_suggestions(
    received_suggestions: list[StaticAnalysisSuggestion],
    expected_issues: list[EvalItemOutput],
    model: str,
) -> list[ModelEvaluationOutput]:
    """
    Uses an LLM to evaluate if the actual bugs are in the suggestions.
    """

    bugs = [
        f"<bug idx={i}>\n{bug.description}\n<location>{bug.encoded_location}</location></bug>"
        for i, bug in enumerate(expected_issues)
    ]

    suggestions = [
        f"<suggestion idx={i}>\n{suggestion.to_text_format()}</suggestion>"
        for i, suggestion in enumerate(received_suggestions)
    ]

    prompt = f"""
    You are an expert at judging code changes and code quality.
    You are given a list of actual bugs in the codebase, and a list of suggestions made by an AI model.

    <goal>
    Score how well an AI model predicts bugs in the codebase, given we know the actual bugs.
    Follow the reasoning rules below to score the model.
    </goal>

    <reasoning_rules>
    When evaluating the actual bugs in the suggestions, consider:
    1. WHAT the bug is, and what the suggestion says it is should be similar and clear for a developer reading the suggestion.
    2. Focus on the core concepts that cause the bug
    </reasoning_rules>

    <output_format>
    For each actual bug, pair the bug idx with the suggestion idx that best matches the bug.
    If no suggestion is a good match, pair the bug idx with -1.

    Include a short reasoning for each pair.
    Add your score based on the reasoning rules above to each pair.
    </output_format>

    <actual_bugs>
    {bugs}
    </actual_bugs>

    <suggested_bugs>
    {suggestions}
    </suggested_bugs>
    """

    response = LlmClient().generate_structured(
        model=GeminiProvider.model(model),
        prompt=prompt,
        response_format=ModelEvaluationOutputList,
    )

    if response.parsed is None:
        raise ValueError("Failed to parse response")

    return response.parsed.evaluations


def evaluate_location_match(
    actual_bug: EvalItemOutput,
    matched_suggestion: StaticAnalysisSuggestion,
) -> bool:
    """
    Evaluates if the actual bug is in the suggestion.
    """
    suggestion_location = f"{matched_suggestion.path}:{matched_suggestion.line}"
    return actual_bug.encoded_location == suggestion_location
