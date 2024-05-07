from typing import Any

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.pipeline import PipelineContext, PipelineStep, PipelineStepTaskRequest
from seer.automation.steps import (
    ParallelizedChainStep,
    ParallelizedChainStepRequest,
    ParallelizedConditionalStep,
)
from seer.automation.utils import make_done_signal


class AutofixPipelineStep(PipelineStep):
    context: AutofixContext

    @classmethod
    def _instantiate_context(cls, request: PipelineStepTaskRequest) -> PipelineContext:
        return AutofixContext.from_run_id(request.run_id)

    def _post_invoke(self, result: Any):
        with self.context.state.update() as cur:
            cur.signals.append(make_done_signal(self.request.step_id))


@celery_app.task()
def autofix_parallelized_conditional_step_task(*args, request: Any):
    AutofixParallelizedConditionalStep(request).invoke()


class AutofixParallelizedConditionalStep(AutofixPipelineStep, ParallelizedConditionalStep):
    name = "AutofixParallelizedConditionalStep"

    @staticmethod
    def get_task():
        return autofix_parallelized_conditional_step_task


@celery_app.task()
def autofix_parallelized_chain_step_task(*args, request: Any):
    AutofixParallelizedChainStep(request).invoke()


class AutofixParallelizedChainStep(AutofixPipelineStep, ParallelizedChainStep):
    name = "AutofixParallelizedChainStep"

    @staticmethod
    def _get_conditional_step_class() -> type[ParallelizedConditionalStep]:
        return AutofixParallelizedConditionalStep

    @staticmethod
    def get_task():
        return autofix_parallelized_chain_step_task

    @staticmethod
    def _instantiate_request(data: dict[str, Any]) -> ParallelizedChainStepRequest:
        return ParallelizedChainStepRequest.model_validate(data)

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()
