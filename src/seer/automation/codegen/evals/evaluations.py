import logging
from unittest import mock

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.codegen.bug_prediction_step import BugPredictionStep
from seer.automation.codegen.evals.models import (
    EvalItemInput,
    EvalItemOutput,
    ModelEvaluationOutput,
    ModelEvaluationOutputList,
)
from seer.automation.codegen.models import (
    BugPrediction,
    CodeFetchIssuesOutput,
    StaticAnalysisSuggestion,
)
from seer.automation.codegen.relevant_warnings_step import RelevantWarningsStep
from seer.automation.codegen.tasks import (
    create_initial_bug_prediction_run,
    create_initial_relevant_warnings_run,
)
from seer.automation.state import DbStateRunTypes

logger = logging.getLogger(__name__)


@observe(name="[Relevant Warnings Eval] Sync run evaluation on item")
def sync_run_evaluation_on_item(
    item: DatasetItemClient,
) -> list[BugPrediction] | None:
    """
    Actually executes the BugPredictionStep with input from the EvalItem.
    It mocks the following parts of the pipeline:
      - Fetching the PR diff
      - Fetching the Sentry issues
    """

    item = EvalItemInput.model_validate(item.input)

    # Build the request from the item.
    request = item.get_request()
    # Make sure we don't post to Overwatch.
    request.should_post_to_overwatch = False

    # Create a proper state for the evaluation run
    # state = create_initial_relevant_warnings_run(request)
    state = create_initial_bug_prediction_run(request)
    run_id = state.get().run_id

    # relevant_warnings_step = RelevantWarningsStep(
    #     request={"run_id": run_id, **request.model_dump()},
    #     type=DbStateRunTypes.RELEVANT_WARNINGS,
    # )

    bug_predictor_step = BugPredictionStep(
        request={"run_id": run_id, **request.model_dump()},
        type=DbStateRunTypes.BUG_PREDICTION,
    )

    # # Mock FilterIssuesComponent to return the issues
    # # Ignoring the filename_to_issues part of the output.
    # mock.patch(
    #     "seer.automation.codegen.relevant_warnings_step.FetchIssuesComponent.invoke",
    #     return_value=CodeFetchIssuesOutput(filename_to_issues={"all_issues": item.issues or []}),
    # )

    # Invoke the step.
    bug_predictor_step.invoke()

    # Grab the suggestions from the state.
    state = bug_predictor_step.context.state
    # return state.get().static_analysis_suggestions
    return state.get().bug_predictions


@observe(name="[Relevant Warnings Eval] Evaluate semantic similarity")
def evaluate_bug_predictions(
    received_bug_predictions: list[BugPrediction],
    # received_suggestions: list[StaticAnalysisSuggestion],
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
        f"<suggestion idx={i}>\n{bug_prediction.to_text_format()}</suggestion>"
        for i, bug_prediction in enumerate(received_bug_predictions)
    ]

    prompt = f"""
    You are an expert at judging code changes and code quality.
    You are given a list of actual bugs in the codebase, and a list of suggestions made by an AI model.

    <goal>
    Score how well an AI model predicts bugs in the codebase, given we know the actual bugs.
    Follow the reasoning rules below to score the model.
    </goal>

    <reasoning_rules>
    When evaluating the suggested bugs, consider:
    1. is the suggestion a _good match_ for an actual bug?
      - Focus on matching the content of the suggestion with the actual bug.
    2. is this suggestion accurately describing the actual bug it was matched to?
      - Focus on the core concepts that cause the bug.
      - The suggestion should lead a developer reading it to be able to find the actual bug.
      - If the content is not related to an actual bug, the suggestion should not be matched.
    </reasoning_rules>

    <output_format>
    For each suggestion, pair it with the bug idx that best matches the suggestion, or -1 if it doesn't match any actual bug.

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
