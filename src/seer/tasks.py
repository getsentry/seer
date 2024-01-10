from typing import Any

import sentry_sdk
from pydantic import BaseModel

from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.types import AutofixRequest
from seer.celery import app as celery_app


class TaskStatusRequest(BaseModel):
    task_id: str


@celery_app.task(time_limit=60 * 20)  # 20 minutes
def run_autofix(data: dict[str, Any]) -> dict[str, Any]:
    with sentry_sdk.start_span(
        op="seer.automation.autofix",
        description="Run autofix on an issue within celery task",
    ) as span:
        autofix = Autofix(AutofixRequest(**data))
        autofix_output = autofix.run()
        span.set_tag("autofix_success", autofix_output is not None)
        return autofix_output.model_dump()
