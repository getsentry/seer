import logging
import os
from typing import Any

import sentry_sdk
from pydantic import BaseModel

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.types import AutofixRequest
from seer.rpc import SentryRpcClient

logger = logging.getLogger("autofix")


class TaskStatusRequest(BaseModel):
    task_id: str


@celery_app.task(time_limit=60 * 20)  # 20 minutes
def run_autofix(data: dict[str, Any]) -> None:
    base_url = os.environ.get("SENTRY_BASE_URL")
    if not base_url:
        raise RuntimeError("SENTRY_BASE_URL must be set")

    client = SentryRpcClient(base_url)

    with sentry_sdk.start_span(
        op="seer.automation.autofix",
        description="Run autofix on an issue within celery task",
    ):
        request = AutofixRequest(**data)
        autofix = Autofix(request, client)
        autofix.run()
