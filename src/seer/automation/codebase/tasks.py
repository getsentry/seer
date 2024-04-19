import logging
from typing import Any

import sentry_sdk

from celery_app.app import app as celery_app
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import CreateCodebaseTaskRequest, UpdateCodebaseTaskRequest
from seer.automation.utils import get_embedding_model

logger = logging.getLogger("autofix")


@celery_app.task(time_limit=60 * 60)  # 1h task timeout
def create_codebase_index(data: dict[str, Any]) -> None:
    request = CreateCodebaseTaskRequest.model_validate(data)

    logger.info("Creating codebase index for repo: %s", request.repo.full_name)

    with sentry_sdk.start_span(
        op="seer.automation.background.create_codebase_index",
        description="Create codebase index within celery task in the background",
    ):
        CodebaseIndex.create(
            request.organization_id,
            request.project_id,
            request.repo,
            embedding_model=get_embedding_model(),
        )

    logger.info("Codebase index created for repo: %s", request.repo.full_name)


@celery_app.task(time_limit=60 * 30)  # 30m task timeout
def update_codebase_index(data: dict[str, Any]) -> None:
    request = UpdateCodebaseTaskRequest.model_validate(data)
    logger.info("Updating codebase index for repo: %s", request.repo_id)

    codebase = CodebaseIndex.from_repo_id(request.repo_id, embedding_model=get_embedding_model())

    with sentry_sdk.start_span(
        op="seer.automation.background.update_codebase_index",
        description="Update codebase index within celery task in the background",
    ):
        codebase.update()

    logger.info("Codebase index updated for repo: %s", request.repo_id)
