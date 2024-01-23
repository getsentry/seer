import logging
import os
from typing import Any

import sentry_sdk
from pydantic import BaseModel

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.types import AutofixRequest, AutofixResult
from seer.rpc import RPCClient

logger = logging.getLogger("autofix")


class TaskStatusRequest(BaseModel):
    task_id: str


@celery_app.task(time_limit=60 * 20)  # 20 minutes
def run_autofix(data: dict[str, Any]) -> None:
    try:
        with sentry_sdk.start_span(
            op="seer.automation.autofix",
            description="Run autofix on an issue within celery task",
        ) as span:
            autofix = Autofix(AutofixRequest(**data))
            autofix_output = autofix.run()
            span.set_tag("autofix_success", autofix_output is not None)

            status = "SUCCESS" if autofix_output is not None else "NOFIX"
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(e)

        autofix_output = None
        status = "ERROR"

    base_url = os.environ.get("SENTRY_BASE_URL")
    if not base_url:
        raise RuntimeError("SENTRY_BASE_URL must be set")

    client = RPCClient(base_url)

    autofix_result = AutofixResult(
        issue_id=data["issue"]["id"], result=autofix_output, status=status
    )
    client.call("autofix_callback", result=autofix_result.model_dump())
