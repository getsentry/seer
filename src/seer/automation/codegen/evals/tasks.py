import logging
import random

from langfuse import Langfuse
from langfuse.api.resources.commons.errors import NotFoundError

from celery_app.app import celery_app
from seer.automation.autofix.evaluations import make_score_name
from seer.automation.codegen.evals.evaluations import (
    evaluate_suggestions,
    sync_run_evaluation_on_item,
)
from seer.automation.codegen.evals.models import (
    CodegenRelevantWarningsEvaluationRequest,
    CodegenRelevantWarningsEvaluationSummary,
    EvalItemOutput,
)
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


@inject
def run_relevant_warnings_evaluation(
    request: CodegenRelevantWarningsEvaluationRequest, app_config: AppConfig = injected
):
    langfuse = Langfuse()

    dataset = langfuse.get_dataset(request.dataset_name)
    items = dataset.items

    if request.run_on_item_id:
        items = [item for item in items if item.id == request.run_on_item_id]

    if request.test and not request.run_on_item_id:
        if request.random_for_test:
            items = random.sample(items, 1)
        else:
            items = items[:1]

    logger.info(
        f"Starting relevant warnings evaluation for dataset {request.dataset_name} with run name '{request.run_name}'."
    )
    logger.info(f"Number of items: {len(items)}")
    logger.info(f"Total number of runs: {len(items) * request.n_runs_per_item}")

    task_ids = []

    for i, item in enumerate(items):
        # Note: This will add ALL the dataset item runs into the CPU queue.
        # As we are not going to be running this in prod yet, it's fine to leave as is.
        # If we do decide to run in prod, should find a way to not overwhelm the CPU queue.
        for _ in range(request.n_runs_per_item):
            async_result = run_relevant_warnings_evaluation_on_item.apply_async(
                (),
                dict(
                    item_id=item.id,
                    run_name=request.run_name,
                    run_description=request.run_description,
                    item_index=i,
                    item_count=len(items),
                ),
                queue=app_config.CELERY_WORKER_QUEUE,
            )
            task_ids.append(async_result.id)

    return CodegenRelevantWarningsEvaluationSummary(
        started=True,
        item_count=len(items),
        task_ids=task_ids,
    )


@celery_app.task()
def run_relevant_warnings_evaluation_on_item(
    *,
    item_id: str,
    run_name: str,
    run_description: str,
    item_index: int,
    item_count: int,
):
    """
    Runs the relevant warnings evaluation on a single item of the dataset.
    It mocks the following parts of the pipeline:
      - Fetching the PR diff
      - Fetching the Sentry issues
    """
    langfuse = Langfuse()

    # This can fail with LangfuseNotFoundError if the item is not found or if it's not active.
    try:
        dataset_item = langfuse.get_dataset_item(item_id)
    except NotFoundError:
        logger.error(f"Item {item_id} not found or not active. Skipping scoring.")
        return

    logger.info(
        f"Starting relevant warnings evaluation for item {item_id}, number {item_index}/{item_count}, with run name '{run_name}'."
    )

    scoring_n_panel = 1
    scoring_model = "gemini-2.5-pro-preview-03-25"

    dataset_item_trace_id = None
    with dataset_item.observe(run_name=run_name, run_description=run_description) as trace_id:
        dataset_item_trace_id = trace_id
        try:
            suggestions = sync_run_evaluation_on_item(dataset_item, langfuse_session_id=trace_id)  # type: ignore
            langfuse.score(
                trace_id=dataset_item_trace_id,
                name="error_running_evaluation",
                value=0,
            )
        except Exception as e:
            logger.exception(f"Error running evaluation: {e}")
            langfuse.score(
                trace_id=dataset_item_trace_id,
                name="error_running_evaluation",
                value=1,
            )
            suggestions = None

    # If suggestions is None we assume no suggestions were generated.
    # rather than an error happening.
    # TODO: Is this the best way to handle this?
    suggestions = suggestions or []

    if not dataset_item.expected_output:
        # This happens in negative test cases where we don't have any expected issues.
        # The value is `{}` in this case.
        list_of_issues = []
    elif isinstance(dataset_item.expected_output, list):
        list_of_issues = dataset_item.expected_output
    else:
        list_of_issues = [dataset_item.expected_output]
    list_of_issues = [EvalItemOutput.model_validate(issue) for issue in list_of_issues]

    scores = evaluate_suggestions(suggestions, list_of_issues, scoring_model)
    valid_scores = [score for score in scores if score.bug_matched_idx != -1]
    suggestions_matched = [score.suggestion_idx for score in valid_scores]
    suggestions_not_matched = [i for i in range(len(suggestions)) if i not in suggestions_matched]
    bugs_matched = [score.bug_matched_idx for score in valid_scores]
    bugs_not_found = [i for i in range(len(list_of_issues)) if i not in bugs_matched]
    scores_content = [score.match_score for score in valid_scores]
    location_match = [score.location_match for score in valid_scores]
    langfuse.score(
        comment=f"Expected number of bugs: {len(list_of_issues)}; Actual bugs found: {[ (suggestion_idx, bug_idx) for suggestion_idx, bug_idx in zip(suggestions_matched, bugs_matched)]}",
        trace_id=dataset_item_trace_id,
        name=make_score_name(model=scoring_model, n_panel=scoring_n_panel, name="bugs_found_count"),
        value=len(bugs_matched),
    )
    langfuse.score(
        comment=f"Individual bug location matches: {location_match}",
        trace_id=dataset_item_trace_id,
        name=make_score_name(model=scoring_model, n_panel=scoring_n_panel, name="location_match"),
        value=sum(location_match),
    )
    langfuse.score(
        comment=f"Individual bug content matches: {scores_content}",
        trace_id=dataset_item_trace_id,
        name=make_score_name(model=scoring_model, n_panel=scoring_n_panel, name="content_match"),
        value=sum(scores_content),
    )
    langfuse.score(
        comment=f"Bugs not found: {bugs_not_found}",
        trace_id=dataset_item_trace_id,
        name=make_score_name(model=scoring_model, n_panel=scoring_n_panel, name="bugs_not_found"),
        value=len(bugs_not_found),
    )
    langfuse.score(
        comment=f"Suggestions not matched to any bug: {suggestions_not_matched}",
        trace_id=dataset_item_trace_id,
        name=make_score_name(model=scoring_model, n_panel=scoring_n_panel, name="noise"),
        value=len(suggestions_not_matched),
    )
