import abc
import logging
from functools import cached_property
from typing import Generic, TypeVar

from pydantic import BaseModel

from seer.automation.models import PromptXmlModel
from seer.automation.pipeline import PipelineContext
from seer.utils import prefix_logger

logger = logging.getLogger(__name__)


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

    @cached_property
    def logger(self):
        run_id = self.context.run_id
        name = f"{type(self).__module__}.{type(self).__qualname__}"
        prefix = f"[{name}] "
        return prefix_logger(prefix, logger, extra={"run_id": run_id})
