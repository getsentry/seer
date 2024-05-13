import abc
import logging
import uuid
from typing import Any

from celery import Task, signature
from pydantic import BaseModel, Field

from seer.automation.state import State
from seer.automation.utils import automation_logger

Signature = Any
SerializedSignature = str

DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS = 50  # 50 seconds
DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS = 60  # 60 seconds


class PipelineContext(abc.ABC):
    state: State
    signals: list[str]

    def __init__(self, state: State):
        self.state = state

    @property
    @abc.abstractmethod
    def run_id(self) -> int:
        pass


class PipelineStepTaskRequest(BaseModel):
    run_id: int
    step_id: int = Field(default_factory=lambda: uuid.uuid4().int)


class PipelineStep(abc.ABC):
    """
    A step in the automation pipeline, complete with the context, request, logging + error handling utils.
    Main method that is run is _invoke, which should be implemented by the subclass.
    """

    name = "PipelineStep"

    def __init__(self, request: dict[str, Any]):
        self.request = self._instantiate_request(request)
        self.context = self._instantiate_context(self.request)

    def invoke(self) -> Any:
        try:
            if not self._pre_invoke():
                return
            result = self._invoke(**self._get_extra_invoke_kwargs())
            self._post_invoke(result)
            return result
        except Exception as e:
            self._handle_exception(e)
            raise e

    def _get_extra_invoke_kwargs(self) -> dict[str, Any]:
        return {}

    def _pre_invoke(self) -> bool:
        return True

    def _post_invoke(self, result: Any) -> Any:
        pass

    @property
    def logger(self):
        name = self.name

        class PipelineLoggingAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                return f"[{name}] {msg}", kwargs

        return PipelineLoggingAdapter(automation_logger)

    @staticmethod
    @abc.abstractmethod
    def get_task() -> Task:
        pass

    @classmethod
    def get_signature(cls, request: PipelineStepTaskRequest, **kwargs) -> Signature:
        return cls.get_task().signature(
            kwargs={"request": request.model_dump(mode="json")}, **kwargs
        )

    @staticmethod
    def instantiate_signature(serialized_signature: SerializedSignature | Signature) -> Signature:
        return signature(serialized_signature)

    @staticmethod
    @abc.abstractmethod
    def _instantiate_request(request: dict[str, Any]) -> PipelineStepTaskRequest:
        pass

    @staticmethod
    @abc.abstractmethod
    def _instantiate_context(request: PipelineStepTaskRequest) -> PipelineContext:
        pass

    @abc.abstractmethod
    def _handle_exception(self, exception: Exception):
        pass

    @abc.abstractmethod
    def _invoke(self, **kwargs) -> Any:
        pass


class PipelineChain(abc.ABC):
    """
    Combine this with PipelineStep to make a step into a chain, which can call other steps.
    """

    def next(self, sig: SerializedSignature | Signature, **apply_async_kwargs):
        signature(sig).apply_async(**apply_async_kwargs)
