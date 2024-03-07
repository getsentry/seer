import abc
import contextlib
import dataclasses
import functools
from typing import Any, ContextManager, Generic, Iterator, TypeVar

from celery import Task
from pydantic import BaseModel

_State = TypeVar("_State", bound=BaseModel)


class State(abc.ABC, Generic[_State]):
    """
    An abstract state buffer that attempts to push state changes to a sink.
    No guarantees are made about the durability or timing of the writes, other than
    from the perspective of the State object other than read-after-write guarantees.
    """

    @abc.abstractmethod
    def get(self) -> _State:
        pass

    @abc.abstractmethod
    def set(self, value: _State):
        pass

    def _update(self) -> Iterator[_State]:
        val = self.get()
        yield val
        self.set(val)

    update = contextlib.contextmanager(_update)


@dataclasses.dataclass
class LocalMemoryState(State[_State]):
    def get(self) -> _State:
        return self.val

    def set(self, value: _State):
        self.val = value

    val: _State


@functools.total_ordering
class TestMemoryState(State[_State]):
    values: list[_State] = dataclasses.field(default_factory=list)

    def __init__(self, initial: _State):
        self.values.append(initial)

    def get(self) -> _State:
        return self.values[-1]

    def set(self, value: _State):
        self.values.append(value)

    def __eq__(self, other: Any) -> bool:
        return self.values[-1] == other

    def __lt__(self, other: Any) -> bool:
        return self.values[-1] < other

    def __hash__(self) -> int:
        return hash(self.values[-1])

    def __contains__(self, other: Any) -> bool:
        return other in self.values

    def __iter__(self) -> Iterator[_State]:
        return iter(self.values)


@dataclasses.dataclass
class CeleryProgressState(State[_State]):
    val: _State
    bind: Task

    def get(self) -> _State:
        return self.val

    def set(self, value: _State):
        self.val = value
        self.bind.update_state("PROGRESS", meta={"value": value.model_dump()})
