import abc
from typing import Any

from seer.automation.state import State


class PipelineContext(abc.ABC):
    state: State

    def __init__(self, state: State):
        self.state = state


class Pipeline(abc.ABC):
    context: PipelineContext

    def __init__(self, context: PipelineContext):
        self.context = context

    @abc.abstractmethod
    def invoke(self, request: Any) -> Any:
        pass
