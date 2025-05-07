import os
import threading
import time
from typing import Any

import sentry_sdk

from celery_app.app import celery_app
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.pipeline import (
    DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS,
    DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS,
    PipelineChain,
    PipelineContext,
    PipelineStep,
    PipelineStepTaskRequest,
)
from seer.automation.state import DbStateRunTypes
from seer.automation.steps import (
    ParallelizedChainConditionalStep,
    ParallelizedChainStep,
    ParallelizedChainStepRequest,
)
from seer.automation.utils import (
    make_done_signal,
    make_kill_signal,
    make_retry_prefix,
    make_retry_signal,
)


class AutofixPipelineStep(PipelineChain, PipelineStep):
    context: AutofixContext

    # Default to no retries, child classes will override this.
    max_retries: int = 0

    thread: threading.Thread | None = None
    thread_kill: bool = False

    def get_retry_count(self) -> int:
        return sum(
            1
            for signal in self.context.signals
            if signal.startswith(make_retry_prefix(self.request.step_id))
        )

    @staticmethod
    def _instantiate_context(
        request: PipelineStepTaskRequest, _: DbStateRunTypes | None = None
    ) -> PipelineContext:
        return AutofixContext.from_run_id(request.run_id)

    def _invoke(self, **kwargs: Any) -> Any:
        state = self.context.state.get()
        autofix_request = state.request

        sentry_sdk.set_tags(
            {
                "organization_id": self.context.organization_id,
                "org_slug": self.context.get_org_slug(self.context.organization_id),
                "project": self.context.project_id,
                "run_id": self.context.state.get().run_id,
                "num_repos": len(autofix_request.repos),
                "has_trace_tree": autofix_request.trace_tree is not None
                and len(autofix_request.trace_tree.events) > 1,
                "has_profile": autofix_request.profile is not None
                and len(autofix_request.profile.execution_tree) > 0,
                "has_issue_summary": autofix_request.issue_summary is not None,
                "has_custom_instruction": bool(autofix_request.instruction),
            }
        )

        user = autofix_request.invoking_user
        if user:
            sentry_sdk.set_user({"id": user.id, "username": user.display_name})

        super()._invoke(**kwargs)

    def _pre_invoke(self) -> bool:
        # Don't run the step instance if it's already been run
        if make_done_signal(self.request.step_id) not in self.context.state.get().signals:
            self.thread = threading.Thread(target=self._check_for_kill)
            self.thread.start()
            return True
        return False

    def _cleanup(self):
        if self.thread:
            self.thread_kill = True
        return super()._cleanup()

    def _get_extra_invoke_kwargs(self) -> dict[str, Any]:
        try:
            cur = self.context.state.get()

            group_id = cur.request.issue.id
            group_short_id = cur.request.issue.short_id
            invoking_user = cur.request.invoking_user

            org_slug = self.context.get_org_slug(cur.request.organization_id)

            tags = {
                "run_id": cur.run_id,
                "org_id": cur.request.organization_id,
                "project_id": cur.request.project_id,
                "group_id": group_id,
                "codebase_indexing": False,
            }
            repo_tags = [f"repo:{repo.full_name}" for repo in cur.request.repos]
            repo_tags_dict = {tag: 1 for tag in repo_tags}
            metadata = {
                "run_id": cur.run_id,
                "org_slug": org_slug,
                "group": {"id": group_id, "short_id": group_short_id},
                "invoking_user": invoking_user,
            }
            langfuse_tags = [
                f"{key}:{value}" for key, value in tags.items() if value is not None
            ] + repo_tags

            return {
                "langfuse_tags": langfuse_tags,
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
            signal = make_done_signal(self.request.step_id)
            cur.signals.append(signal)

    def _check_for_kill(self):
        while True:
            if self.thread_kill:
                return
            kill_signal = make_kill_signal()
            if kill_signal in self.context.state.get().signals:
                with self.context.state.update() as cur:
                    cur.signals.remove(kill_signal)
                self.thread_kill = True
                os._exit(1)  # kills the thread and the whole step
            time.sleep(0.1)

    def _handle_exception(self, exception: Exception):
        retries = self.get_retry_count()
        if self.max_retries > retries:
            new_retry_index = retries + 1
            self.logger.info(
                f"Retrying {self.request.step_id}, {new_retry_index}/{self.max_retries} times"
            )
            self.context.event_manager.on_error(str(exception), should_completely_error=False)

            with self.context.state.update() as cur:
                cur.signals.append(make_retry_signal(self.request.step_id, new_retry_index))

            self.request.is_retry = True
            self.next(self.get_signature(self.request))
        else:
            self.logger.error(
                f"Failed to run {self.request.step_id} after {self.max_retries} retries"
            )

            # This time this will error the entire pipeline.
            self.context.event_manager.on_error(
                "Oops, something went wrong inside. We use Sentry too, so we're already on it."
            )


@celery_app.task(
    time_limit=DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS,
    soft_time_limit=DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS,
)
def autofix_parallelized_conditional_step_task(*args, request: Any):
    AutofixParallelizedChainConditionalStep(request).invoke()


class AutofixParallelizedChainConditionalStep(
    ParallelizedChainConditionalStep, AutofixPipelineStep
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


class AutofixParallelizedChainStep(ParallelizedChainStep, AutofixPipelineStep):
    name = "AutofixParallelizedChainStep"

    @staticmethod
    def get_task():
        return autofix_parallelized_chain_step_task

    @staticmethod
    def _get_conditional_step_class() -> type[ParallelizedChainConditionalStep]:
        return AutofixParallelizedChainConditionalStep

    @staticmethod
    def _instantiate_request(data: dict[str, Any]) -> ParallelizedChainStepRequest:
        return ParallelizedChainStepRequest.model_validate(data)

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error(str(exception))
