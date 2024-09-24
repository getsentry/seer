import datetime
import logging
import random
from typing import Literal, cast

import sentry_sdk
from langfuse import Langfuse

from celery_app.app import celery_app
from celery_app.config import CeleryQueues
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.evaluations import (
    RootCauseScoreResult,
    make_score_name,
    score_one,
    score_root_causes,
    sync_run_evaluation_on_item,
    sync_run_execution,
    sync_run_root_cause,
)
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixCreatePrUpdatePayload,
    AutofixEvaluationRequest,
    AutofixRequest,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    AutofixUpdateRequest,
    AutofixUserMessagePayload,
)
from seer.automation.autofix.runs import create_initial_autofix_run
from seer.automation.autofix.state import ContinuationState
from seer.automation.autofix.steps.coding_step import AutofixCodingStep, AutofixCodingStepRequest
from seer.automation.autofix.steps.root_cause_step import RootCauseStep, RootCauseStepRequest
from seer.automation.models import InitializationError
from seer.automation.utils import (
    get_sentry_client,
    process_repo_provider,
    raise_if_no_genai_consent,
)
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session

logger = logging.getLogger(__name__)


def get_autofix_state_from_pr_id(provider: str, pr_id: int) -> ContinuationState | None:
    with Session() as session:
        run_state = (
            session.query(DbRunState)
            .join(DbPrIdToAutofixRunIdMapping, DbPrIdToAutofixRunIdMapping.run_id == DbRunState.id)
            .filter(
                DbPrIdToAutofixRunIdMapping.provider == process_repo_provider(provider),
                DbPrIdToAutofixRunIdMapping.pr_id == pr_id,
            )
            .order_by(DbRunState.id.desc())
            .first()
        )
        if run_state is None:
            return None

        continuation = ContinuationState.from_id(run_state.id, AutofixContinuation)

        return continuation


def get_autofix_state(
    *, group_id: int | None = None, run_id: int | None = None
) -> ContinuationState | None:
    with Session() as session:
        run_state: DbRunState | None = None
        if group_id is not None:
            if run_id is not None:
                raise ValueError("Either group_id or run_id must be provided, not both")
            run_state = (
                session.query(DbRunState)
                .filter(DbRunState.group_id == group_id)
                .order_by(DbRunState.id.desc())
                .first()
            )
        elif run_id is not None:
            run_state = session.query(DbRunState).filter(DbRunState.id == run_id).first()
        else:
            raise ValueError("Either group_id or run_id must be provided")

        if run_state is None:
            return None

        continuation = ContinuationState.from_id(run_state.id, AutofixContinuation)

        return continuation


def get_all_autofix_runs_after(after: datetime.datetime):
    with Session() as session:
        runs = (
            session.query(DbRunState)
            .filter(DbRunState.last_triggered_at > after, DbRunState.type == "autofix")
            .all()
        )
        return [ContinuationState.from_id(run.id, AutofixContinuation) for run in runs]


@celery_app.task(time_limit=15)
def check_and_mark_recent_autofix_runs():
    logger.info("Checking and marking recent autofix runs")
    after = datetime.datetime.now() - datetime.timedelta(hours=1, minutes=15)
    logger.info(f"Getting all autofix runs after {after}")
    runs = get_all_autofix_runs_after(after)
    logger.info(f"Got {len(runs)} runs")
    for run in runs:
        check_and_mark_if_timed_out(run)


def check_and_mark_if_timed_out(state: ContinuationState):
    with state.update() as cur:
        if cur.is_running and cur.has_timed_out:
            cur.mark_running_steps_errored()
            cur.status = AutofixStatus.ERROR

            # Will log to Sentry. We will have an alert set up to notify us when this happens.
            logger.error(f"Autofix run {cur.run_id} has timed out")


def run_autofix_root_cause(
    request: AutofixRequest,
):
    state = create_initial_autofix_run(request)

    cur_state = state.get()

    # Process has no further work.
    if cur_state.status in AutofixStatus.terminal():
        logger.warning(f"Ignoring job, state {cur_state.status}")
        return

    RootCauseStep.get_signature(
        RootCauseStepRequest(
            run_id=cur_state.run_id,
        ),
        queue=CeleryQueues.DEFAULT,
    ).apply_async()

    return cur_state.run_id


def run_autofix_execution(request: AutofixUpdateRequest):
    state = ContinuationState.from_id(request.run_id, model=AutofixContinuation)

    raise_if_no_genai_consent(state.get().request.organization_id)

    with state.update() as cur:
        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    event_manager.send_coding_start()

    payload = cast(AutofixRootCauseUpdatePayload, request.payload)

    try:
        event_manager.set_selected_root_cause(payload)
        cur = state.get()

        # Process has no further work.
        if cur.status in AutofixStatus.terminal():
            logger.warning(f"Ignoring job, state {cur.status}")
            return

        AutofixCodingStep.get_signature(
            AutofixCodingStepRequest(
                run_id=cur.run_id,
            ),
            queue=CeleryQueues.DEFAULT,
        ).apply_async()
    except InitializationError as e:
        sentry_sdk.capture_exception(e)
        raise e


def run_autofix_create_pr(request: AutofixUpdateRequest):
    if not isinstance(request.payload, AutofixCreatePrUpdatePayload):
        raise ValueError("Invalid payload type for create_pr")

    state = ContinuationState.from_id(request.run_id, model=AutofixContinuation)

    raise_if_no_genai_consent(state.get().request.organization_id)

    with state.update() as cur:
        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    context = AutofixContext(
        state=state, sentry_client=get_sentry_client(), event_manager=event_manager
    )

    context.commit_changes(
        repo_external_id=request.payload.repo_external_id, repo_id=request.payload.repo_id
    )


def receive_user_message(request: AutofixUpdateRequest):
    if not isinstance(request.payload, AutofixUserMessagePayload):
        raise ValueError("Invalid payload type for user_message")

    state = ContinuationState.from_id(request.run_id, model=AutofixContinuation)
    with state.update() as cur:
        cur.steps[-1].receive_user_message(request.payload.text)


def run_autofix_evaluation(request: AutofixEvaluationRequest):
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
        f"Starting autofix evaluation for dataset {request.dataset_name} with run name '{request.run_name}'."
    )
    logger.info(f"Number of items: {len(items)}")
    logger.info(f"Total number of runs: {len(items) * request.n_runs_per_item}")

    for i, item in enumerate(items):
        # Note: This will add ALL the dataset item runs into the CPU queue.
        # As we are not going to be running this in prod yet, it's fine to leave as is.
        # If we do decide to run in prod, should find a way to not overwhelm the CPU queue.
        for _ in range(request.n_runs_per_item):
            run_autofix_evaluation_on_item.apply_async(
                (),
                dict(
                    item_id=item.id,
                    run_name=request.run_name,
                    run_description=request.run_description,
                    run_type=request.run_type,
                    item_index=i,
                    item_count=len(items),
                ),
                queue=CeleryQueues.DEFAULT,
            )


@celery_app.task()
def run_autofix_evaluation_on_item(
    *,
    item_id: str,
    run_name: str,
    run_description: str,
    run_type: Literal["execution", "full", "root_cause"],
    item_index: int,
    item_count: int,
):
    langfuse = Langfuse()

    dataset_item = langfuse.get_dataset_item(item_id)

    logger.info(
        f"Starting autofix evaluation for item {item_id}, number {item_index}/{item_count}, with run name '{run_name}'."
    )

    scoring_n_panel = 5
    scoring_model = "gpt-4o-2024-05-13"

    diff: str | None = None

    with dataset_item.observe(run_name=run_name, run_description=run_description) as trace_id:
        if run_type == "root_cause":
            causes: list[RootCauseAnalysisItem] | None = None
            try:
                causes = sync_run_root_cause(dataset_item, langfuse_session_id=trace_id)
            except Exception as e:
                logger.error(f"Error running root cause analysis: {e}")

            if causes:
                root_cause_score: RootCauseScoreResult = score_root_causes(
                    dataset_item,
                    causes,
                    n_panel=scoring_n_panel,
                    model=scoring_model,
                    langfuse_session_id=trace_id,
                )

                # Will output 4 scores for a run item:
                # - `"highest_score"`: The score for the highest scored cause out of all the returned root causes.
                # - `"positioning_score"`: Positioning of the highest scored cause, if the highest scored cause is first, this score is `1.0`. If it is last, it will be `0.0`. The score is calculated proportional to the number of causes provided.
                # - `"mean_score"`: The mean score of all the root causes.
                # - `"error_weighted_score"` This score is the same as the highest score but scored 0 if there is an error or no root cause returned. This is used to weight the score in the aggregated run result.

                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=root_cause_score.get("highest_score"),
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="highest_score"
                    ),
                    value=root_cause_score.get("highest_score"),
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="positioning_score"
                    ),
                    value=root_cause_score.get("position_score"),
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="mean_score"
                    ),
                    value=root_cause_score.get("mean_score"),
                )
            else:
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=0,
                )
        elif run_type == "execution":
            try:
                diff = sync_run_execution(dataset_item, langfuse_session_id=trace_id)
            except Exception as e:
                logger.error(f"Error running evaluation: {e}")

            if diff:
                score = score_one(
                    dataset_item,
                    diff,
                    n_panel=scoring_n_panel,
                    model=scoring_model,
                    langfuse_session_id=trace_id,
                )

                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=score,
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="score"
                    ),
                    value=score,
                )
            else:
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=0,
                )
        else:
            try:
                diff = sync_run_evaluation_on_item(
                    dataset_item,
                    scoring_n_panel=scoring_n_panel,
                    scoring_model=scoring_model,
                    trace_id=trace_id,
                    langfuse_session_id=trace_id,
                )
            except Exception as e:
                logger.error(f"Error running evaluation: {e}")

            if diff:
                score = score_one(
                    dataset_item,
                    diff,
                    n_panel=scoring_n_panel,
                    model=scoring_model,
                    langfuse_session_id=trace_id,
                )

                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=score,
                )
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="score"
                    ),
                    value=score,
                )
            else:
                langfuse.score(
                    trace_id=trace_id,
                    name=make_score_name(
                        model=scoring_model, n_panel=scoring_n_panel, name="error_weighted_score"
                    ),
                    value=0,
                )
