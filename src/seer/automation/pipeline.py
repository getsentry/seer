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
