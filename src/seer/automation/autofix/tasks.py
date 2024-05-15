import logging
from typing import cast

import sentry_sdk

from celery_app.config import CeleryQueues
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixCreatePrUpdatePayload,
    AutofixRequest,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    AutofixUpdateRequest,
    CustomRootCauseSelection,
    SuggestedFixRootCauseSelection,
)
from seer.automation.autofix.state import ContinuationState
from seer.automation.autofix.steps.create_missing_indexes_chain import (
    CreateAnyMissingCodebaseIndexesStepRequest,
    CreateMissingIndexesStep,
)
from seer.automation.autofix.steps.execution_step import (
    AutofixExecutionStep,
    AutofixExecutionStepRequest,
)
from seer.automation.autofix.steps.root_cause_step import RootCauseStep, RootCauseStepRequest
from seer.automation.autofix.utils import get_sentry_client
from seer.automation.models import InitializationError
from seer.automation.utils import process_repo_provider
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


def get_autofix_state(group_id: int) -> ContinuationState | None:
    with Session() as session:
        run_state = (
            session.query(DbRunState)
            .filter(DbRunState.group_id == group_id)
            .order_by(DbRunState.id.desc())
            .first()
        )
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
    state = ContinuationState.new(
        AutofixContinuation(request=request),
        group_id=request.issue.id,
    )

    with state.update() as cur:
        cur.mark_triggered()
    cur = state.get()

    event_manager = AutofixEventManager(state)
    event_manager.send_root_cause_analysis_start()

    # Process has no further work.
    if cur.status in AutofixStatus.terminal():
        logger.warning(f"Ignoring job, state {cur.status}")
        return

    CreateMissingIndexesStep.get_signature(
        CreateAnyMissingCodebaseIndexesStepRequest(
            run_id=cur.run_id,
            next=RootCauseStep.get_signature(
                RootCauseStepRequest(
                    run_id=cur.run_id,
                ),
                queue=CeleryQueues.DEFAULT,
            ),
        ),
        queue=CeleryQueues.DEFAULT,
    ).apply_async()


def run_autofix_execution(request: AutofixUpdateRequest):
    state = ContinuationState.from_id(request.run_id, model=AutofixContinuation)

    with state.update() as cur:
        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    event_manager.send_planning_start()

    payload = cast(AutofixRootCauseUpdatePayload, request.payload)

    try:
        root_cause: CustomRootCauseSelection | SuggestedFixRootCauseSelection | None = None
        if payload.custom_root_cause:
            root_cause = CustomRootCauseSelection(
                custom_root_cause=payload.custom_root_cause,
            )
        elif payload.cause_id is not None and payload.fix_id is not None:
            root_cause = SuggestedFixRootCauseSelection(
                cause_id=payload.cause_id,
                fix_id=payload.fix_id,
            )

        if root_cause is None:
            raise ValueError("Invalid root cause update payload")

        event_manager.set_selected_root_cause(root_cause)
        cur = state.get()

        # Process has no further work.
        if cur.status in AutofixStatus.terminal():
            logger.warning(f"Ignoring job, state {cur.status}")
            return

        CreateMissingIndexesStep.get_signature(
            CreateAnyMissingCodebaseIndexesStepRequest(
                run_id=cur.run_id,
                next=AutofixExecutionStep.get_signature(
                    AutofixExecutionStepRequest(
                        run_id=cur.run_id,
                    ),
                    queue=CeleryQueues.CUDA,
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

    context.commit_changes(repo_id=request.payload.repo_id)

    event_manager.send_pr_creation_complete()
