import logging
from typing import Any

import sentry_sdk
from aiohttp import ClientResponseError
from aiohttp.http_exceptions import HttpProcessingError
from celery import Task

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixCompleteArgs,
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    AutofixStepUpdateArgs,
)
from seer.automation.models import InitializationError
from seer.automation.state import CeleryProgressState
from seer.db import ProcessRequest
from seer.rpc import RpcClient, SentryRpcClient
from seer.task_factory import AsyncTaskFactory, async_task_factory

logger = logging.getLogger("autofix")


@async_task_factory
class AutofixTaskFactory(AsyncTaskFactory):
    def matches(self, process_request: ProcessRequest) -> bool:
        return process_request.name.startswith("autofix:")

    async def invoke(self, process_request: ProcessRequest):
        rpc_client: RpcClient = SentryRpcClient()
        try:
            autofix_group_state = await rpc_client.acall(
                "get_autofix_state", issue_id=process_request.payload["issue"]["id"]
            )
        except ClientResponseError as e:
            sentry_sdk.capture_exception(e)
            # The group does not exist anymore, let us ignore this issue.
            if e.status == 404:
                return

        async for update in self.async_celery_job(
            lambda: run_autofix.delay(process_request.payload, autofix_group_state)
        ):
            continuation = AutofixContinuation.model_validate(update["value"])

            if continuation.status in {AutofixStatus.ERROR, AutofixStatus.COMPLETED}:
                await rpc_client.acall(
                    "on_autofix_complete",
                    **AutofixCompleteArgs(
                        issue_id=continuation.request.issue.id,
                        status=continuation.status,
                        steps=continuation.steps,
                        fix=continuation.fix,
                    ).model_dump(mode="json"),
                )
            else:
                await rpc_client.acall(
                    "on_autofix_step_update",
                    **AutofixStepUpdateArgs(
                        issue_id=continuation.request.issue.id,
                        status=continuation.status,
                        steps=continuation.steps,
                    ).model_dump(mode="json"),
                )


@celery_app.task(bind=True, time_limit=60 * 60 * 5)  # 5 hour task timeout
def run_autofix(
    self: Task,
    request_data: dict[str, Any],
    autofix_group_state: dict[str, Any] | None,
):
    if autofix_group_state is None:
        autofix_group_state = {}
    continuation = AutofixContinuation.model_validate(
        dict(
            request=request_data,
            **autofix_group_state,
        )
    )

    # Process has no further work.
    if continuation.status in AutofixStatus.terminal():
        return

    state = CeleryProgressState(val=continuation, bind=self)

    request = AutofixRequest(**request_data)
    event_manager = AutofixEventManager(state)
    try:
        if request.has_timed_out:
            raise InitializationError("Timeout while dealing with autofix request.")

        with sentry_sdk.start_span(
            op="seer.automation.autofix",
            description="Run autofix on an issue within celery task",
        ):
            autofix = Autofix(request, event_manager)
            autofix.run()
    except InitializationError as e:
        event_manager.send_autofix_complete(None)
        sentry_sdk.capture_exception(e)
        raise e
