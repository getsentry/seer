import abc
from typing import Generic, TypeVar

from pydantic import BaseModel

from seer.automation.models import PromptXmlModel
from seer.automation.pipeline import PipelineContext
from seer.automation.state import State


class BaseComponentRequest(BaseModel):
    pass


class BaseComponentXmlOutput(PromptXmlModel):
    pass


class BaseComponentOutput(BaseModel):
    pass


BCR = TypeVar("BCR", bound=BaseComponentRequest)
BCO = TypeVar("BCO", bound=BaseComponentOutput | BaseComponentXmlOutput)


class BaseComponent(abc.ABC, Generic[BCR, BCO]):
    context: PipelineContext

    def __init__(self, context: PipelineContext):
        self.context = context

    @abc.abstractmethod
    def invoke(self, request: BCR) -> BCO | None:
        pass
