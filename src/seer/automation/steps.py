import abc
from typing import Any, Optional, Type

from celery import signature

from celery_app.config import CeleryQueues
from seer.automation.pipeline import PipelineChain, PipelineStep, PipelineStepTaskRequest, Signature
from seer.automation.utils import make_done_signal


class ConditionalStepRequest(PipelineStepTaskRequest):
    on_success: Optional[Signature] = None
    on_failure: Optional[Signature] = None


class ConditionalStep(PipelineStep):
    """
    Utility conditional step with a condition that determines whether to run the on_success or on_failure steps.
    """

    request: ConditionalStepRequest

    @abc.abstractmethod
    def condition(self) -> bool:
        pass

    def _invoke(self):
        result = self.condition()
        self.logger.debug(f"Conditional step {self.request.step_id} condition result: {result}")

        if self.condition():
            if self.request.on_success:
                signature(self.request.on_success).apply_async()
        else:
            if self.request.on_failure:
                signature(self.request.on_failure).apply_async()

    def _handle_exception(self, exception: Exception):
        pass


class ParallelizedChainConditionalStepRequest(ConditionalStepRequest):
    expected_signals: list[str]


class ParallelizedChainConditionalStep(ConditionalStep):
    """
    The conditional for the ParallelizedChainStep. It checks if all expected signals are present in the context.
    """

    name = "ParallelizedChainConditionalStep"
    request: ParallelizedChainConditionalStepRequest

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> ParallelizedChainConditionalStepRequest:
        return ParallelizedChainConditionalStepRequest(**request)

    def condition(self):
        result = all(signal in self.context.signals for signal in self.request.expected_signals)

        return result


class ParallelizedChainStepRequest(PipelineStepTaskRequest):
    steps: list[Signature]
    on_success: Optional[Signature]


class ParallelizedChainStep(PipelineChain, PipelineStep):
    """
    Runs multiple steps in parallel and waits for all of them to complete successfully before continuing.
    """

    name = "ParallelizedChainStep"
    request: ParallelizedChainStepRequest

    @staticmethod
    @abc.abstractmethod
    def _get_conditional_step_class() -> Type[ParallelizedChainConditionalStep]:
        pass

    def _invoke(self):
        signatures = [signature(step) for step in self.request.steps]

        expected_signals = [
            make_done_signal(sig.kwargs["request"]["step_id"]) for sig in signatures
        ]

        self.logger.debug(f"Running {len(signatures)} parallelized steps.")

        for sig in signatures:
            sig.apply_async(
                link=self._get_conditional_step_class().get_signature(
                    ParallelizedChainConditionalStepRequest(
                        run_id=self.context.run_id,
                        expected_signals=expected_signals,
                        on_success=self.request.on_success,
                    ),
                    queue=CeleryQueues.DEFAULT,
                )
            )
