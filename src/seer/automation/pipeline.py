import abc
import logging
import uuid
from typing import Any, Type

from celery import Task, signature
from pydantic import BaseModel, Field

from celery_app.config import CeleryQueues
from seer.automation.state import State
from seer.automation.utils import automation_logger

Signature = Any


class PipelineContext(abc.ABC):
    state: State
    signals: list[str]

    def __init__(self, state: State):
        self.state = state

    @property
    @abc.abstractmethod
    def run_id(self) -> int:
        pass


class PipelineSideEffect(abc.ABC):
    context: PipelineContext

    @abc.abstractmethod
    def invoke(self):
        pass


class Pipeline(abc.ABC):
    context: PipelineContext
    side_effects: list[PipelineSideEffect] = []

    def __init__(self, context: PipelineContext):
        self.context = context

    def _invoke_side_effects(self):
        for side_effect in self.side_effects:
            side_effect.invoke()

    def invoke(self) -> Any:
        try:
            self._invoke_side_effects()
            return self._invoke()
        except Exception as e:
            self._handle_exception(e)
            raise e

    @abc.abstractmethod
    def _handle_exception(self, exception: Exception):
        pass

    @abc.abstractmethod
    def _invoke(self) -> Any:
        pass


class PipelineStepTaskRequest(BaseModel):
    run_id: int
    step_id: int = Field(default_factory=lambda: uuid.uuid4().int)


class PipelineStep(abc.ABC):
    name = "PipelineStep"

    def __init__(self, request: dict[str, Any]):
        self.request = self._instantiate_request(request)
        self.context = self._instantiate_context(self.request)

    def invoke(self) -> Any:
        try:
            result = self._invoke()
            self._post_invoke(result)
            return result
        except Exception as e:
            self._handle_exception(e)
            raise e

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

    @classmethod
    @abc.abstractmethod
    def _instantiate_request(cls, request: dict[str, Any]) -> PipelineStepTaskRequest:
        pass

    @classmethod
    @abc.abstractmethod
    def _instantiate_context(cls, request: PipelineStepTaskRequest) -> PipelineContext:
        pass

    @abc.abstractmethod
    def _handle_exception(self, exception: Exception):
        pass

    @abc.abstractmethod
    def _invoke(self) -> Any:
        pass


class PipelineChain(abc.ABC):
    def next(
        self,
        sig: Any,
    ):
        signature(sig).apply_async()
