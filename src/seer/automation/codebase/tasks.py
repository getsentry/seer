import logging
from typing import Any

import sentry_sdk

from celery_app.app import app as celery_app
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import (
    CodebaseIndexStatus,
    CodebaseNamespaceStatus,
    IndexNamespaceTaskRequest,
    UpdateCodebaseTaskRequest,
)
from seer.automation.codebase.namespace import CodebaseNamespaceManager
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import RepoDefinition
from seer.automation.utils import get_embedding_model

logger = logging.getLogger("autofix")


def create_codebase_index(
    organization_id: int,
    project_id: int,
    repo: RepoDefinition,
) -> int:
    namespace_id = CodebaseIndex.create(organization_id, project_id, repo)

    return namespace_id


@celery_app.task(time_limit=60 * 60)  # 1h task timeout
def index_namespace(data: dict[str, Any]) -> None:
    request = IndexNamespaceTaskRequest.model_validate(data)

    with sentry_sdk.start_span(
        op="seer.automation.background.create_codebase_index",
        description="Create codebase index within celery task in the background",
    ):
        CodebaseIndex.index(
            namespace_id=request.namespace_id,
            embedding_model=get_embedding_model(),
        )


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


def check_repo_access(
    repo: RepoDefinition,
) -> bool:
    return RepoClient.check_repo_access(repo)


def get_codebase_index_status(
    organization_id: int,
    project_id: int,
    repo: RepoDefinition,
    sha: str | None = None,
    tracking_branch: str | None = None,
) -> CodebaseIndexStatus:
    namespace = CodebaseNamespaceManager.get_namespace(
        organization_id, project_id, repo, sha, tracking_branch
    )

    if namespace:
        if namespace.status == CodebaseNamespaceStatus.PENDING:
            return CodebaseIndexStatus.INDEXING

        return CodebaseIndexStatus.UP_TO_DATE

    return CodebaseIndexStatus.NOT_INDEXED
