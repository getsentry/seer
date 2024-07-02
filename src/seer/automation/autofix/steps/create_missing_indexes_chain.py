from typing import Any

import sentry_sdk
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

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

    @observe(name="Autofix - Create Missing Indices Step")
    @ai_track(description="Autofix - Create Missing Indices Step")
    def _invoke(self, **kwargs):
        event_details = EventDetails.from_event(self.context.state.get().request.issue.events[0])

        repos_to_create: list[RepoDefinition] = []
        repos_to_update: list[int] = []

        for repo in self.context.repos:
            codebase = self.context.codebases.get(repo.external_id)

            # If a codebase is not ready delete it and recreate it.
            if codebase:
                ready = codebase.workspace.is_ready()
                integrity = ready and codebase.verify_file_integrity()

                if not integrity:
                    # Log integrity failures for now
                    log = f"Codebase workspace integrity check failed for repo: {repo.full_name}."
                    self.logger.debug(log)
                    sentry_sdk.capture_message(log)

                # TODO: Delete codebase if integrity check fails too
                if not ready:
                    log = (
                        f"Codebase workspace was not ready for repo: {repo.full_name}, recreating."
                    )
                    self.logger.debug(log)
                    sentry_sdk.capture_message(log)

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
                                ).model_dump(mode="json"),
                            ),
                            countdown=60 * 15,  # 15 minutes
                            queue=CeleryQueues.CUDA,
                        )

        steps: list[Signature] = []

        for repo in repos_to_create:
            steps.append(
                CreateIndexStep.get_signature(
                    CodebaseIndexingStepRequest(**self.step_request_fields, repo=repo),
                    queue=CeleryQueues.CUDA,
                )
            )

        for repo_id in repos_to_update:
            steps.append(
                UpdateIndexStep.get_signature(
                    UpdateIndexStepRequest(**self.step_request_fields, repo_id=repo_id),
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
                        **self.step_request_fields,
                        steps=steps,
                        on_success=self.request.next,
                    ),
                    queue=CeleryQueues.DEFAULT,
                ),
            )
        else:
            self.next(self.request.next)
