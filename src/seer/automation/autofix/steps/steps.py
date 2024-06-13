from typing import Any

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.pipeline import (
    DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS,
    DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS,
    PipelineContext,
    PipelineStep,
    PipelineStepTaskRequest,
)
from seer.automation.steps import (
    ParallelizedChainConditionalStep,
    ParallelizedChainStep,
    ParallelizedChainStepRequest,
)
from seer.automation.utils import make_done_signal


class AutofixPipelineStep(PipelineStep):
    context: AutofixContext

    @staticmethod
    def _instantiate_context(request: PipelineStepTaskRequest) -> PipelineContext:
        return AutofixContext.from_run_id(request.run_id)

    def _pre_invoke(self) -> bool:
        # Don't run the step instance if it's already been run
        return make_done_signal(self.request.step_id) not in self.context.state.get().signals

    def _get_extra_invoke_kwargs(self) -> dict[str, Any]:
        try:
            cur = self.context.state.get()

            group_id = cur.request.issue.id
            group_short_id = cur.request.issue.short_id
            invoking_user = cur.request.invoking_user
            codebases = [
                self._get_codebase_metadata(codebase.repo_id) for codebase in cur.codebases.values()
            ]

            org_slug = self.context.get_org_slug(cur.request.organization_id)

            tags = {
                "run_id": cur.run_id,
                "org_id": cur.request.organization_id,
                "project_id": cur.request.project_id,
                "group_id": group_id,
            }
            repo_tags = [f"repo:{codebase.get('external_slug')}" for codebase in codebases]
            repo_tags_dict = {tag: 1 for tag in repo_tags}
            metadata = {
                "run_id": cur.run_id,
                "org_slug": org_slug,
                "group": {"id": group_id, "short_id": group_short_id},
                "invoking_user": invoking_user,
                "codebases": codebases,
            }

            return {
                "langfuse_tags": [
                    f"{key}:{value}" for key, value in tags.items() if value is not None
                ]
                + repo_tags,
                "langfuse_metadata": metadata,
                "langfuse_session_id": str(cur.run_id),
                "langfuse_user_id": f"org:{org_slug}" if org_slug else None,
                "sentry_tags": {
                    **tags,
                    **repo_tags_dict,
                },
                "sentry_data": metadata,
            }
        except Exception:
            return {}

    def _post_invoke(self, result: Any):
        with self.context.state.update() as cur:
            cur.signals.append(make_done_signal(self.request.step_id))

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()

    def _get_codebase_metadata(self, repo_id: int) -> dict[str, Any]:
        codebase = self.context.get_codebase(repo_id)
        if codebase:
            return {
                "repo_id": codebase.repo_info.id,
                "namespace_id": codebase.namespace.id,
                "external_id": codebase.repo_info.external_id,
                "external_slug": codebase.repo_info.external_slug,
                "namespace_status": codebase.namespace.status,
                "namespace_tracking_branch": codebase.namespace.tracking_branch,
                "sha": codebase.namespace.sha,
            }
        return {}


@celery_app.task(
    time_limit=DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS,
    soft_time_limit=DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS,
)
def autofix_parallelized_conditional_step_task(*args, request: Any):
    AutofixParallelizedChainConditionalStep(request).invoke()


class AutofixParallelizedChainConditionalStep(
    AutofixPipelineStep, ParallelizedChainConditionalStep
):
    name = "AutofixParallelizedChainConditionalStep"

    @staticmethod
    def get_task():
        return autofix_parallelized_conditional_step_task


@celery_app.task(
    time_limit=DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS,
    soft_time_limit=DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS,
)
def autofix_parallelized_chain_step_task(*args, request: Any):
    AutofixParallelizedChainStep(request).invoke()


class AutofixParallelizedChainStep(AutofixPipelineStep, ParallelizedChainStep):
    name = "AutofixParallelizedChainStep"

    @staticmethod
    def _get_conditional_step_class() -> type[ParallelizedChainConditionalStep]:
        return AutofixParallelizedChainConditionalStep

    @staticmethod
    def get_task():
        return autofix_parallelized_chain_step_task

    @staticmethod
    def _instantiate_request(data: dict[str, Any]) -> ParallelizedChainStepRequest:
        return ParallelizedChainStepRequest.model_validate(data)

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()
