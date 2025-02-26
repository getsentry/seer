import abc
import logging
import uuid
from functools import cached_property
from typing import Any, Generic, TypeVar

from celery import Task, signature
from pydantic import BaseModel, Field

from seer.automation.state import DbStateRunTypes, State
from seer.utils import prefix_logger

logger = logging.getLogger(__name__)

Signature = Any
SerializedSignature = Any

DEFAULT_PIPELINE_STEP_SOFT_TIME_LIMIT_SECS = 50  # 50 seconds
DEFAULT_PIPELINE_STEP_HARD_TIME_LIMIT_SECS = 60  # 60 seconds

PIPELINE_SYNC_SIGNAL = "pipeline_run_mode:sync"


class PipelineContext(abc.ABC):
    state: State

    def __init__(self, state: State):
        self.state = state

    @property
    @abc.abstractmethod
    def run_id(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def signals(self) -> list[str]:
        pass

    @signals.setter
    @abc.abstractmethod
    def signals(self, signals: list[str]):
        pass


class PipelineStepTaskRequest(BaseModel):
    run_id: int
    step_id: int = Field(default_factory=lambda: uuid.uuid4().int)


def make_step_request_fields(context: PipelineContext):
    return {"run_id": context.run_id}


# Define a type variable that is bound to PipelineStepTaskRequest
_RequestType = TypeVar("_RequestType", bound=PipelineStepTaskRequest)
_ContextType = TypeVar("_ContextType", bound=PipelineContext)


class PipelineStep(abc.ABC, Generic[_RequestType, _ContextType]):
    """
    A step in the automation pipeline, complete with the context, request, logging + error handling utils.
    Main method that is run is _invoke, which should be implemented by the subclass.
    """

    name = "PipelineStep"
    request: _RequestType
    context: _ContextType

    def __init__(self, request: dict[str, Any], type: DbStateRunTypes | None = None):
        self.request = self._instantiate_request(request)
        self.context = self._instantiate_context(self.request, type)

    def invoke(self) -> Any:
        try:
            if not self._pre_invoke():
                self._cleanup()
                return
            result = self._invoke(**self._get_extra_invoke_kwargs())
            self._post_invoke(result)
            return result
        except Exception as e:
            self._handle_exception(e)
            raise e
        finally:
            self._cleanup()

    def _get_extra_invoke_kwargs(self) -> dict[str, Any]:
        return {}

    def _pre_invoke(self) -> bool:
        return True

    def _post_invoke(self, result: Any) -> Any:
        pass

    def _cleanup(self):
        pass

    @cached_property
    def logger(self):
        run_id = self.context.run_id
        name = self.name
        prefix = f"[{run_id=}] [{name}] "
        return prefix_logger(prefix, logger)

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
    def _instantiate_request(request: dict[str, Any]) -> _RequestType:
        pass

    @staticmethod
    @abc.abstractmethod
    def _instantiate_context(
        request: PipelineStepTaskRequest, type: DbStateRunTypes | None = None
    ) -> _ContextType:
        pass

    @abc.abstractmethod
    def _handle_exception(self, exception: Exception):
        pass

    @abc.abstractmethod
    def _invoke(self, **kwargs) -> Any:
        pass

    @property
    def step_request_fields(self):
        return make_step_request_fields(self.context)


class PipelineChain(PipelineStep):
    """
    A PipelineStep which can call other steps.
    """

    def next(self, sig: SerializedSignature | Signature, **apply_async_kwargs):
        if PIPELINE_SYNC_SIGNAL in self.context.signals:
            signature(sig).apply(**apply_async_kwargs)
        else:
            signature(sig).apply_async(**apply_async_kwargs)
