from typing import Any

import sentry_sdk

from celery_app.app import app as celery_app
from celery_app.config import CeleryQueues
from seer.automation.autofix.steps.create_index_step import (
    CodebaseIndexingStepRequest,
    CreateIndexStep,
)
from seer.automation.autofix.steps.steps import AutofixParallelizedChainStep, AutofixPipelineStep
from seer.automation.autofix.steps.update_index_step import UpdateIndexStep, UpdateIndexStepRequest
from seer.automation.codebase.models import UpdateCodebaseTaskRequest
from seer.automation.codebase.tasks import update_codebase_index
from seer.automation.models import EventDetails, RepoDefinition
from seer.automation.pipeline import (
    DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS,
    DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS,
    PipelineChain,
    PipelineStepTaskRequest,
    Signature,
)
from seer.automation.steps import ParallelizedChainStepRequest


@celery_app.task(
    time_limit=DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS,
    soft_time_limit=DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS,
)
def create_missing_indexes_task(*args, request: Any):
    CreateMissingIndexesStep(request).invoke()


class CreateAnyMissingCodebaseIndexesStepRequest(PipelineStepTaskRequest):
    next: Signature


class CreateMissingIndexesStep(PipelineChain, AutofixPipelineStep):
    name = "CreateMissingIndexesStep"
    request: CreateAnyMissingCodebaseIndexesStepRequest

    @staticmethod
    def _instantiate_request(data: dict[str, Any]) -> CreateAnyMissingCodebaseIndexesStepRequest:
        return CreateAnyMissingCodebaseIndexesStepRequest.model_validate(data)

    @staticmethod
    def get_task():
        return create_missing_indexes_task

    def _invoke(self):
        event_details = EventDetails.from_event(self.context.state.get().request.issue.events[0])

        repos_to_create: list[RepoDefinition] = []
        repos_to_update: list[int] = []

        for repo in self.context.repos:
            codebase = self.context.get_codebase_from_external_id(repo.external_id)

            # If a codebase is not ready, delete it and recreate it.
            if codebase and not codebase.workspace.is_ready():
                sentry_sdk.capture_message(
                    f"Codebase workspace was not ready for repo: {repo.full_name}, recreating"
                )
                codebase.workspace.delete()
                codebase = None

            if not codebase:
                repos_to_create.append(repo)
            else:
                self.context.event_manager.add_log(
                    f"Codebase index ready for repo: {repo.full_name}"
                )
                if codebase.is_behind():
                    if codebase.diff_contains_stacktrace_files(event_details):
                        # Update now
                        repos_to_update.append(codebase.repo_info.id)
                    else:
                        # Update later
                        update_codebase_index.apply_async(
                            (
                                UpdateCodebaseTaskRequest(
                                    repo_id=codebase.repo_info.id,
                                ),
                            ),
                            countdown=60 * 15,  # 15 minutes
                            queue=CeleryQueues.CUDA,
                        )

        steps: list[Signature] = []

        for repo in repos_to_create:
            steps.append(
                CreateIndexStep.get_signature(
                    CodebaseIndexingStepRequest(run_id=self.context.run_id, repo=repo),
                    queue=CeleryQueues.CUDA,
                )
            )

        for repo_id in repos_to_update:
            steps.append(
                UpdateIndexStep.get_signature(
                    UpdateIndexStepRequest(run_id=self.context.run_id, repo_id=repo_id),
                    queue=CeleryQueues.CUDA,
                )
            )

        self.logger.info(
            f"Creating {len(repos_to_create)} and updating {len(repos_to_update)} codebase indexes"
        )

        if steps:
            self.next(
                AutofixParallelizedChainStep.get_signature(
                    ParallelizedChainStepRequest(
                        run_id=self.context.run_id,
                        steps=steps,
                        on_success=self.request.next,
                    ),
                    queue=CeleryQueues.DEFAULT,
                ),
            )
        else:
            self.next(self.request.next)
