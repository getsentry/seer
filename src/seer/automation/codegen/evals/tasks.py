import logging
import random

from langfuse import Langfuse

from celery_app.app import celery_app
from seer.automation.autofix.evaluations import make_score_name
from seer.automation.codegen.evals.evaluations import (
    score_suggestions_content,
    score_suggestions_length,
    sync_run_evaluation_on_item,
)
from seer.automation.codegen.evals.models import (
    CodegenRelevantWarningsEvaluationRequest,
    CodegenRelevantWarningsEvaluationSummary,
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
        with item.observe(run_name=request.run_name, run_description=request.run_description):
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

    dataset_item = langfuse.get_dataset_item(item_id)

    logger.info(
        f"Starting relevant warnings evaluation for item {item_id}, number {item_index}/{item_count}, with run name '{run_name}'."
    )

    scoring_n_panel = 5  # do we even need this?
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

    length_score = score_suggestions_length(suggestions, dataset_item)
    langfuse.score(
        trace_id=dataset_item_trace_id,
        name="length_of_solutions_score",
        value=length_score,
    )

    suggestions_content_score = score_suggestions_content(suggestions, dataset_item, scoring_model)
    langfuse.score(
        trace_id=dataset_item_trace_id,
        name=make_score_name(
            model=scoring_model, n_panel=scoring_n_panel, name="suggestions_content_score"
        ),
        value=suggestions_content_score,
    )
