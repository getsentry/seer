import abc
from typing import Any

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

    def invoke_side_effects(self):
        for side_effect in self.side_effects:
            side_effect.invoke()

    @abc.abstractmethod
    def invoke(self, request: Any) -> Any:
        pass
