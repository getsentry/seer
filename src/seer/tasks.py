import os
from typing import Any

import sentry_sdk
from pydantic import BaseModel

from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.types import AutofixRequest, AutofixResult
from seer.celery_app import app as celery_app
from seer.rpc import RPCClient


class TaskStatusRequest(BaseModel):
    task_id: str


@celery_app.task(time_limit=60 * 20)  # 20 minutes
def run_autofix(data: dict[str, Any]) -> None:
    with sentry_sdk.start_span(
        op="seer.automation.autofix",
        description="Run autofix on an issue within celery task",
    ) as span:
        autofix = Autofix(AutofixRequest(**data))
        autofix_output = autofix.run()
        span.set_tag("autofix_success", autofix_output is not None)

    base_url = os.environ.get("SENTRY_BASE_URL")
    if not base_url:
        raise RuntimeError("SENTRY_BASE_URL must be set")

    client = RPCClient(base_url)

    autofix_result = AutofixResult(
        issue_id=data["issue"]["id"],
        result=autofix_output,
    )
    client.call("autofix_callback", result=autofix_result.model_dump())
