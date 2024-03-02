import logging
import os
from typing import Any

import sentry_sdk

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixRequest
from seer.automation.models import InitializationError
from seer.rpc import RpcClient, SentryRpcClient
from seer.tasks import TaskFactory, async_task_factory

logger = logging.getLogger("autofix")


@celery_app.task(time_limit=60 * 60 * 5)  # 5 hour task timeout
def run_autofix(
    request_data: dict[str, Any], process_request_name: str = "", *, client: RpcClient | None = None
) -> None:
    if client is None:
        client = SentryRpcClient(os.environ["SENTRY_BASE_URL"])
    # client = DummyRpcClient()

    request = AutofixRequest(**request_data)
    event_manager = AutofixEventManager(client, request.issue.id)
    try:
        with sentry_sdk.start_span(
            op="seer.automation.autofix",
            description="Run autofix on an issue within celery task",
        ):
            autofix = Autofix(request, event_manager)
            autofix.run()
    except InitializationError as e:
        event_manager.mark_running_steps_errored()
        event_manager.send_autofix_complete(None)
        sentry_sdk.capture_exception(e)


@async_task_factory
class CodebaseIndexCreator(TaskFactory):
    pass
