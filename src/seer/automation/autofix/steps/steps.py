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

    def _post_invoke(self, result: Any):
        with self.context.state.update() as cur:
            cur.signals.append(make_done_signal(self.request.step_id))

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()


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
