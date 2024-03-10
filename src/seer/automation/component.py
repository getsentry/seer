import abc
from typing import Generic, TypeVar

from pydantic import BaseModel

from seer.automation.pipeline import PipelineContext
from seer.automation.state import State


class BaseComponentRequest(BaseModel):
    pass


class BaseComponentOutput(BaseModel):
    pass


BCR = TypeVar("BCR", bound=BaseComponentRequest)


class BaseComponent(abc.ABC, Generic[BCR]):
    context: PipelineContext

    def __init__(self, context: PipelineContext):
        self.context = context

    @abc.abstractmethod
    def invoke(self, request: BCR) -> BaseComponentOutput | None:
        pass
