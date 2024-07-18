from typing import Any

import sentry_sdk
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app as celery_app
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.config import (
    AUTOFIX_CREATE_INDEX_HARD_TIME_LIMIT_SECS,
    AUTOFIX_CREATE_INDEX_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest


class CodebaseIndexingStepRequest(PipelineStepTaskRequest):
    repo: RepoDefinition


@celery_app.task(
    time_limit=AUTOFIX_CREATE_INDEX_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_CREATE_INDEX_SOFT_TIME_LIMIT_SECS,
)
def create_index_task(*args, request: Any):
    CreateIndexStep(request).invoke()


class CreateIndexStep(AutofixPipelineStep):
    name = "CreateIndexStep"

    request: CodebaseIndexingStepRequest
    context: AutofixContext

    @staticmethod
    def _instantiate_request(data: dict[str, Any]) -> CodebaseIndexingStepRequest:
        return CodebaseIndexingStepRequest.model_validate(data)

    @staticmethod
    def get_task():
        return create_index_task

    @observe(name="Autofix - Create Index Step")
    @ai_track(description="Autofix - Create Index Step")
    def _invoke(self, **kwargs):
        repo = self.request.repo

        self.context.event_manager.send_codebase_indexing_start()  # This should only create the step once and get every next time it's called...

        self.context.event_manager.add_log(f"Creating codebase index for repo: {repo.full_name}")
        with sentry_sdk.start_span(
            op="seer.automation.autofix.codebase_index.create",
            description="Create codebase index",
        ) as span:
            span.set_tag("repo", repo.full_name)
            self.context.create_codebase_index(repo)
        self.context.event_manager.add_log(f"Created codebase index for repo: {repo.full_name}")
