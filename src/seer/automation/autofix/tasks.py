import logging
from typing import cast

import sentry_sdk
from langfuse import Langfuse

from celery_app.app import app as celery_app
from celery_app.config import CeleryQueues
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.evaluations import score_one, sync_run_evaluation_on_item
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixCreatePrUpdatePayload,
    AutofixRequest,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    AutofixUpdateRequest,
)
from seer.automation.autofix.runs import create_initial_autofix_run
from seer.automation.autofix.state import ContinuationState
from seer.automation.autofix.steps.create_missing_indexes_chain import (
    CreateAnyMissingCodebaseIndexesStepRequest,
    CreateMissingIndexesStep,
)
from seer.automation.autofix.steps.planning_chain import (
    AutofixPlanningStep,
    AutofixPlanningStepRequest,
)
from seer.automation.autofix.steps.root_cause_step import RootCauseStep, RootCauseStepRequest
from seer.automation.models import InitializationError
from seer.automation.utils import (
    get_sentry_client,
    process_repo_provider,
    raise_if_no_genai_consent,
)
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session

logger = logging.getLogger("autofix")


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


def check_and_mark_if_timed_out(state: ContinuationState):
    with state.update() as cur:
        if cur.has_timed_out:
            cur.mark_running_steps_errored()
            cur.status = AutofixStatus.ERROR


def run_autofix_root_cause(
    request: AutofixRequest,
):
    state = create_initial_autofix_run(request)

    cur_state = state.get()

    # Process has no further work.
    if cur_state.status in AutofixStatus.terminal():
        logger.warning(f"Ignoring job, state {cur_state.status}")
        return

    if request.options.disable_codebase_indexing:
        RootCauseStep.get_signature(
            RootCauseStepRequest(
                run_id=cur_state.run_id,
            ),
            queue=CeleryQueues.DEFAULT,
        ).apply_async()
    else:
        CreateMissingIndexesStep.get_signature(
            CreateAnyMissingCodebaseIndexesStepRequest(
                run_id=cur_state.run_id,
                next=RootCauseStep.get_signature(
                    RootCauseStepRequest(
                        run_id=cur_state.run_id,
                    ),
                    queue=CeleryQueues.DEFAULT,
                ),
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
    event_manager.send_planning_start()

    payload = cast(AutofixRootCauseUpdatePayload, request.payload)

    try:
        event_manager.set_selected_root_cause(payload)
        cur = state.get()

        # Process has no further work.
        if cur.status in AutofixStatus.terminal():
            logger.warning(f"Ignoring job, state {cur.status}")
            return

        if cur.request.options.disable_codebase_indexing:
            AutofixPlanningStep.get_signature(
                AutofixPlanningStepRequest(
                    run_id=cur.run_id,
                ),
                queue=CeleryQueues.DEFAULT,
            ).apply_async()
        else:
            CreateMissingIndexesStep.get_signature(
                CreateAnyMissingCodebaseIndexesStepRequest(
                    run_id=cur.run_id,
                    next=AutofixPlanningStep.get_signature(
                        AutofixPlanningStepRequest(
                            run_id=cur.run_id,
                        ),
                        queue=CeleryQueues.DEFAULT,
                    ),
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
        state=state,
        sentry_client=get_sentry_client(),
        event_manager=event_manager,
        skip_loading_codebase=True,
    )

    event_manager.send_pr_creation_start()

    context.commit_changes(
        repo_external_id=request.payload.repo_external_id, repo_id=request.payload.repo_id
    )

    event_manager.send_pr_creation_complete()


def run_autofix_evaluation(dataset_name: str, run_name: str):
    langfuse = Langfuse()

    dataset = langfuse.get_dataset(dataset_name)

    logger.info(
        f"Starting autofix evaluation for dataset {dataset_name} with run name '{run_name}'."
    )
    logger.info(f"Number of items: {len(dataset.items)}")

    for i, item in enumerate(dataset.items):
        run_autofix_evaluation_on_item.apply_async(
            (item.id, run_name, i, len(dataset.items)), queue=CeleryQueues.DEFAULT
        )


@celery_app.task()
def run_autofix_evaluation_on_item(item_id: str, run_name: str, item_index: int, item_count: int):
    langfuse = Langfuse()

    dataset_item = langfuse.get_dataset_item(item_id)

    logger.info(
        f"Starting autofix evaluation for item {item_id}, number {item_index}/{item_count}, with run name '{run_name}'."
    )

    with dataset_item.observe(run_name=run_name) as trace_id:
        diff = sync_run_evaluation_on_item(dataset_item)
        if diff:
            langfuse.score(
                trace_id=trace_id,
                name="gpt4_0125_n3_score",
                value=score_one(dataset_item, diff),
            )
