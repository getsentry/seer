from typing import Any

import sentry_sdk
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.config import (
    AUTOFIX_UPDATE_INDEX_HARD_TIME_LIMIT_SECS,
    AUTOFIX_UPDATE_INDEX_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.pipeline import PipelineStepTaskRequest


class UpdateIndexStepRequest(PipelineStepTaskRequest):
    repo_id: int


@celery_app.task(
    time_limit=AUTOFIX_UPDATE_INDEX_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_UPDATE_INDEX_SOFT_TIME_LIMIT_SECS,
)
def update_index_task(*args, request: Any):
    UpdateIndexStep(request).invoke()


class UpdateIndexStep(AutofixPipelineStep):
    name = "UpdateIndexStep"

    request: UpdateIndexStepRequest
    context: AutofixContext

    @staticmethod
    def _instantiate_request(data: dict[str, Any]) -> UpdateIndexStepRequest:
        return UpdateIndexStepRequest.model_validate(data)

    @staticmethod
    def get_task():
        return update_index_task

    @observe(name="Autofix – Change Describer Step")
    @ai_track(description="Autofix - Change Describer Step")
    def _invoke(self, **kwargs):
        codebase = self.context.get_codebase_from_repo_id(self.request.repo_id)

        if not codebase:
            raise ValueError(f"Codebase index for repo_id {self.request.repo_id} not found")

        self.logger.info(
            f"Waiting for codebase index update for repo {codebase.repo_info.external_slug}"
        )
        self.context.event_manager.send_codebase_indexing_start()
        self.context.event_manager.add_log(
            f"Updating codebase index for repo: {codebase.repo_info.external_slug}"
        )
        with sentry_sdk.start_span(
            op="seer.automation.autofix.codebase_index.update",
            description="Update codebase index",
        ) as span:
            span.set_tag("repo", codebase.repo_info.external_slug)
            codebase.update()
        self.logger.info("Codebase index updated")
        self.context.event_manager.add_log(
            f"Updated codebase index for repo: {codebase.repo_info.external_slug}"
        )
