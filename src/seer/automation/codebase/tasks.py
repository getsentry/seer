import logging
from typing import Any

import sentry_sdk

from celery_app.app import app as celery_app
from celery_app.models import UpdateCodebaseTaskRequest
from seer.automation.codebase.codebase_index import CodebaseIndex

logger = logging.getLogger("autofix")


@celery_app.task(time_limit=60 * 60)  # 1h task timeout
def update_codebase_index(data: dict[str, Any]) -> None:
    request = UpdateCodebaseTaskRequest(**data)
    logger.info("Updating codebase index for repo: %s", request.repo_id)

    codebase = CodebaseIndex.from_repo_id(request.repo_id)

    with sentry_sdk.start_span(
        op="seer.automation.background.update_codebase_index",
        description="Update codebase index within celery task in the background",
    ):
        codebase.update()

    logger.info("Codebase index updated for repo: %s", request.repo_id)
