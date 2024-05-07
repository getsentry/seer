import abc
from typing import Any, Type

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.pipeline import PipelineContext, PipelineStep, PipelineStepTaskRequest


class AutofixPipelineStep(PipelineStep):
    context: AutofixContext

    @staticmethod
    @abc.abstractmethod
    def _get_request_class() -> Type[PipelineStepTaskRequest]:
        pass

    @classmethod
    def _instantiate_request(cls, request: dict[str, Any]) -> PipelineStepTaskRequest:
        return cls._get_request_class().model_validate(request)

    @classmethod
    def _instantiate_context(cls, request: PipelineStepTaskRequest) -> PipelineContext:
        return AutofixContext.from_run_id(request.run_id)
