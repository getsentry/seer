import abc
from typing import Any, Type

from celery import Task
from pydantic import BaseModel

from celery_app.config import CeleryQueues
from seer.automation.state import State


class PipelineContext(abc.ABC):
    state: State

    def __init__(self, state: State):
        self.state = state


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


class PipelineStep(abc.ABC):
    def __init__(self, request: dict[str, Any]):
        self.request = self._instantiate_request(request)
        self.context = self._instantiate_context(self.request)

    def invoke(self) -> Any:
        try:
            return self._invoke()
        except Exception as e:
            self._handle_exception(e)
            raise e

    @staticmethod
    @abc.abstractmethod
    def get_task() -> Task:
        pass

    @classmethod
    def get_signature(cls, request: PipelineStepTaskRequest, **kwargs) -> Any:
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
        signature: Any,
    ):
        signature.apply_async()

    def next_concurrent(
        self,
        runs: list[Any],
    ):
        for run in runs:
            run.apply_async()
