import logging
from unittest import mock

from langfuse.client import DatasetItemClient
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.codebase.models import PrFile
from seer.automation.codegen.models import (
    CodeFetchIssuesOutput,
    CodegenRelevantWarningsRequest,
    StaticAnalysisSuggestion,
)
from seer.automation.codegen.relevant_warnings_step import RelevantWarningsStep
from seer.automation.codegen.tasks import create_initial_relevant_warnings_run
from seer.automation.state import DbStateRunTypes
from seer.automation.utils import extract_text_inside_tags

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

    # Create a proper state for the evaluation run
    state = create_initial_relevant_warnings_run(request)
    run_id = state.get().run_id

    relevant_warnings_step = RelevantWarningsStep(
        request={"run_id": run_id, **request.model_dump()},
        type=DbStateRunTypes.RELEVANT_WARNINGS,
    )
    # Mock the repo client to return the pr_files
    mock_repo_client = mock.Mock()
    mock_repo_client.repo.get_pull.return_value.get_files.return_value = pr_files
    relevant_warnings_step.context.get_repo_client = mock.Mock(return_value=mock_repo_client)
    # Mock FilterIssuesComponent to return the issues
    # Ignoring the filename_to_issues part of the output.
    mock_fetch_issues = mock.patch(
        "seer.automation.codegen.relevant_warnings_step.FetchIssuesComponent.invoke",
        return_value=CodeFetchIssuesOutput(filename_to_issues={"all_issues": raw_issues}),
    )
    relevant_warnings_step.context.filter_issues_component = mock_fetch_issues.start()

    # Invoke the step.
    relevant_warnings_step.invoke()

    # Grab the suggestions from the state.
    state = relevant_warnings_step.context.state
    return state.get().static_analysis_suggestions


def score_suggestions_length(
    suggestions: list[StaticAnalysisSuggestion],
    item: DatasetItemClient,
) -> float:
    """
    Scores that the number of suggestions we got is close to the number of expected suggestions.

    """
    expected_suggestions = item.expected_output["suggestions"]
    return abs(len(suggestions) - len(expected_suggestions))


@observe(name="[Relevant Warnings Eval] Score suggestions content")
def score_suggestions_content(
    suggestions: list[StaticAnalysisSuggestion],
    item: DatasetItemClient,
    scoring_model: str,
) -> float:
    """
    Scores the content of static analysis suggestions by comparing them with expected suggestions.

    Evaluation criteria:
    - path and line must match (exact match)
    - short description is similar to expected (LLM evaluation)
    - justification is similar (LLM evaluation)
    - related_warning_id, related_issue_id match (exact match)
    - severity_score and confidence_score are similar (within a threshold)

    Returns a float score from 0 to 1, where 1 means perfect match.
    """
    if not suggestions:
        return 0.0

    expected_suggestions = item.expected_output["suggestions"] or []

    # Convert expected suggestions to StaticAnalysisSuggestion objects
    expected_suggestions_objects = [
        StaticAnalysisSuggestion.model_validate(suggestion) for suggestion in expected_suggestions
    ]

    # Calculate scores for each suggestion
    total_score = 0.0
    matched_suggestions = 0

    for suggestion in suggestions:
        # Find the best matching expected suggestion
        best_match_score = 0.0
        best_match = None

        for expected in expected_suggestions_objects:
            # Check path and line match (exact match)
            if suggestion.path != expected.path or suggestion.line != expected.line:
                continue

            # Check related IDs match
            ids_score = 0.0
            if suggestion.related_warning_id == expected.related_warning_id:
                ids_score += 0.5
            if suggestion.related_issue_id == expected.related_issue_id:
                ids_score += 0.5

            # Check severity and confidence scores are similar (within 0.2 threshold)
            score_diff = abs(suggestion.severity_score - expected.severity_score)
            confidence_diff = abs(suggestion.confidence_score - expected.confidence_score)
            scores_score = (int(score_diff <= 0.2) + int(confidence_diff <= 0.2)) / 2

            # Use LLM to evaluate semantic similarity of descriptions and justifications
            description_score = evaluate_semantic_similarity(
                suggestion.short_description, expected.short_description, scoring_model
            )

            justification_score = evaluate_semantic_similarity(
                suggestion.justification, expected.justification, scoring_model
            )

            # Calculate total score for this match
            match_score = (
                +ids_score * 0.2  # 20% weight
                + scores_score * 0.2  # 20% weight
                + description_score * 0.3  # 30% weight
                + justification_score * 0.3  # 30% weight
            )

            if match_score > best_match_score:
                best_match_score = match_score
                best_match = expected

        # Add the best match score to the total
        if best_match:
            total_score += best_match_score
            matched_suggestions += 1

    # Calculate final score
    if matched_suggestions == 0:
        return 0.0

    # Average score across all matched suggestions
    avg_score = total_score / matched_suggestions

    # Penalize for unmatched suggestions
    match_ratio = matched_suggestions / max(len(suggestions), len(expected_suggestions_objects))

    # Final score is a combination of average match quality and match ratio
    final_score = avg_score * 0.7 + match_ratio * 0.3

    return final_score


# Should we just use embeddings similarity here, instead of a full LLM?
# The embeddings might be enough to encode the semantic similarity (what we want)
# and it would be faster and cheaper.
@observe(name="[Relevant Warnings Eval] Evaluate semantic similarity")
def evaluate_semantic_similarity(received_text: str, expected_text: str, model: str) -> float:
    """
    Uses an LLM to evaluate the semantic similarity between two texts.
    Returns a score from 0 to 1, where 1 means perfect semantic match.
    """

    prompt = f"""
    <goal>
    Evaluate the semantic similarity between two texts. Score from 0 to 1, where 1 means perfect semantic match.
    Make sure to use granular scoring up to 3 decimal places (i.e. 0.45, 0.125, etc).
    </goal>

    <reasoning_rules>
    When evaluating semantic similarity, consider:
    1. Core meaning and concepts
    2. Key technical terms
    3. Context and implications
    4. Ignore minor wording differences
    </reasoning_rules>

    <output_format>
    1. Provide your reasoning inside a <reasoning> tag.
    2. Provide the similarity score inside a <score> tag as a float between 0 and 1.
    </output_format>

    <received_text>
    {received_text}
    </received_text>

    <expected_text>
    {expected_text}
    </expected_text>
    """

    response = LlmClient().generate_text(
        model=GeminiProvider.model(model),
        prompt=prompt,
    )

    if not response.message.content:
        return 0.0

    score_str = extract_text_inside_tags(response.message.content, "score")
    score = float(score_str) if score_str else 0.0

    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))
