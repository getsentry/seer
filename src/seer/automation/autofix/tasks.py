import dataclasses
import logging
from typing import Any, cast

import sentry_sdk

from celery_app.app import app as celery_app
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
from seer.automation.autofix.pipelines import AutofixExecution, AutofixRootCause
from seer.automation.autofix.utils import get_sentry_client
from seer.automation.models import InitializationError
from seer.automation.state import DbState
from seer.db import DbRunState, Session
from seer.rpc import RpcClient

logger = logging.getLogger("autofix")


@dataclasses.dataclass
class ContinuationState(DbState[AutofixContinuation]):
    pass


def get_autofix_state(group_id: int) -> AutofixContinuation | None:
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

        return continuation.get()


@celery_app.task(time_limit=60 * 15)  # 15 minute task timeout
def run_autofix_root_cause(
    request_data: dict[str, Any],
    autofix_group_state: dict[str, Any] | None = None,
):
    request = AutofixRequest.model_validate(request_data)
    state = ContinuationState.new(
        AutofixContinuation(request=request),
        group_id=request.issue.id,
    )
    event_manager = AutofixEventManager(state)
    event_manager.send_root_cause_analysis_pending()
    try:
        cur = state.get()

        # Process has no further work.
        if cur.status in AutofixStatus.terminal():
            logger.warning(f"Ignoring job, state {cur.status}")
            return

        if cur.request.has_timed_out:
            raise InitializationError("Timeout while dealing with autofix request.")

        with sentry_sdk.start_span(
            op="seer.automation.autofix",
            description="Run autofix on an issue within celery task",
        ):
            context = AutofixContext(
                event_manager=event_manager,
                sentry_client=get_sentry_client(),
                state=state,
            )
            autofix_root_cause = AutofixRootCause(context)
            autofix_root_cause.invoke()
    except InitializationError as e:
        sentry_sdk.capture_exception(e)
        raise e


@celery_app.task(time_limit=60 * 15)  # 15 minute task timeout
def run_autofix_execution(
    request_data: dict[str, Any],
    autofix_group_state: dict[str, Any] | None = None,
):
    request = AutofixUpdateRequest.model_validate(request_data)
    state = ContinuationState.from_id(request.run_id, model=AutofixContinuation)
    event_manager = AutofixEventManager(state)
    event_manager.send_planning_pending()

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

        if cur.request.has_timed_out:
            raise InitializationError("Timeout while dealing with autofix request.")

        with sentry_sdk.start_span(
            op="seer.automation.autofix_execution",
            description="Run autofix on an issue within celery task",
        ):
            context = AutofixContext(
                event_manager=event_manager,
                sentry_client=get_sentry_client(),
                state=state,
            )
            autofix_execution = AutofixExecution(context)
            autofix_execution.invoke()
    except InitializationError as e:
        sentry_sdk.capture_exception(e)
        raise e


@celery_app.task(time_limit=60 * 5)  # 5 minute task timeout
def run_autofix_create_pr(request_data: dict[str, Any]):
    request = AutofixUpdateRequest.model_validate(request_data)

    if not isinstance(request.payload, AutofixCreatePrUpdatePayload):
        raise ValueError("Invalid payload type for create_pr")

    state = ContinuationState.from_id(request.run_id, model=AutofixContinuation)
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
