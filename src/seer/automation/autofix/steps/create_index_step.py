from typing import Any, Type

import sentry_sdk

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.autofix.utils import autofix_logger
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineStepTaskRequest


class CodebaseIndexingStepRequest(PipelineStepTaskRequest):
    repo: RepoDefinition


@celery_app.task()
def create_index_task(*args, request: Any):
    CreateIndexStep(request).invoke()


class CreateIndexStep(AutofixPipelineStep):
    request: CodebaseIndexingStepRequest
    context: AutofixContext

    @staticmethod
    def _instantiate_request(data: dict[str, Any]) -> CodebaseIndexingStepRequest:
        return CodebaseIndexingStepRequest.model_validate(data)

    @staticmethod
    def get_task():
        return create_index_task

    def _invoke(self):
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

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()
