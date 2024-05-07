import dataclasses
from typing import Any, Literal, Type

import sentry_sdk

from celery_app.app import app as celery_app
from celery_app.config import CeleryQueues
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.steps.check_indexing_done import (
    CheckIndexingDoneStep,
    CheckIndexingDoneStepRequest,
)
from seer.automation.autofix.steps.create_index_step import (
    CodebaseIndexingStepRequest,
    CreateIndexStep,
)
from seer.automation.autofix.steps.root_cause import RootCauseStep
from seer.automation.autofix.steps.step import AutofixPipelineStep
from seer.automation.models import EventDetails, RepoDefinition
from seer.automation.pipeline import PipelineChain, PipelineStep, PipelineStepTaskRequest


@celery_app.task()
def create_missing_indexes_task(request: Any):
    CreateMissingIndexesStep(request).invoke()


class CreateAnyMissingCodebaseIndexesStepRequest(PipelineStepTaskRequest):
    next_step_name: Literal["RootCauseStep", "ExecutionStep"]


class CreateMissingIndexesStep(PipelineChain, AutofixPipelineStep):
    request_class = CreateAnyMissingCodebaseIndexesStepRequest

    request: CreateAnyMissingCodebaseIndexesStepRequest

    @staticmethod
    def get_task():
        return create_missing_indexes_task

    def _invoke(self):
        event_details = EventDetails.from_event(self.context.state.get().request.issue.events[0])

        repos_to_create: list[RepoDefinition] = []
        repos_to_update: list[RepoDefinition] = []

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
                    f"Codebase index already exists for repo: {repo.full_name}"
                )
                if codebase.is_behind():
                    if codebase.diff_contains_stacktrace_files(event_details):
                        # Update right now and wait
                        # autofix_logger.debug(
                        #     f"Waiting for codebase index update for repo {codebase.repo_info.external_slug}"
                        # )
                        # self.context.event_manager.send_codebase_indexing_start()
                        # self.context.event_manager.add_log(
                        #     f"Creating codebase index for repo: {codebase.repo_info.external_slug}"
                        # )
                        # with sentry_sdk.start_span(
                        #     op="seer.automation.autofix.codebase_index.update",
                        #     description="Update codebase index",
                        # ) as span:
                        #     span.set_tag("repo", codebase.repo_info.external_slug)
                        #     codebase.update()
                        # autofix_logger.debug(f"Codebase index updated")
                        # self.context.event_manager.add_log(
                        #     f"Created codebase index for repo: {codebase.repo_info.external_slug}"
                        # )
                        repos_to_update.append(repo)
                    else:
                        # Update later
                        pass

        steps: list[tuple[Type[PipelineStep], PipelineStepTaskRequest]] = []
        expected_signal_keys = [
            f"codebase_index_done_{repo.external_id}" for repo in repos_to_create
        ] + [f"codebase_index_done_{repo.external_id}" for repo in repos_to_update]

        next_step_signature = None
        if self.request.next_step_name == "RootCauseStep":
            next_step_signature = RootCauseStep.get_signature(
                PipelineStepTaskRequest(run_id=self.context.run_id),
                queue=CeleryQueues.DEFAULT,
            )
        # elif self.request.next_step_name == "ExecutionStep":
        #     next_step_signature = ExecutionStep.task.signature(
        #         PipelineStepTaskRequest(run_id=self.context.run_id),
        #         queue=CeleryQueues.DEFAULT,
        #     )

        check_step_signature = CheckIndexingDoneStep.get_signature(
            CheckIndexingDoneStepRequest(
                run_id=self.context.run_id,
                expected_signal_keys=expected_signal_keys,
                success_link=next_step_signature,
            ),
            queue=CeleryQueues.DEFAULT,
        )

        for repo in repos_to_create:
            steps.append(
                CreateIndexStep.get_signature(
                    CodebaseIndexingStepRequest(
                        run_id=self.context.run_id,
                        repo=repo,
                        done_signal_key=f"codebase_index_done_{repo.external_id}",
                    ),
                    queue=CeleryQueues.CUDA,
                    link=check_step_signature,
                )
            )

        for repo in repos_to_update:
            steps.append(
                CreateIndexStep.get_signature(
                    CodebaseIndexingStepRequest(
                        run_id=self.context.run_id,
                        repo=repo,
                        done_signal_key=f"codebase_index_done_{repo.external_id}",
                    ),
                    queue=CeleryQueues.CUDA,
                    link=check_step_signature,
                )
            )

        if steps:
            self.next_concurrent(steps)
        else:
            self.next(next_step_signature)

    def _handle_exception(self, exception: Exception):
        pass
