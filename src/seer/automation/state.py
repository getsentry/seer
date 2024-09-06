import abc
import contextlib
import dataclasses
import functools
from enum import Enum
from typing import Any, Generic, Iterator, Type, TypeVar, cast

from celery import Task
from pydantic import BaseModel

from seer.db import DbRunState, Session

_State = TypeVar("_State", bound=BaseModel)


class DbStateRunTypes(str, Enum):
    AUTOFIX = "autofix"
    UNIT_TEST = "unit-test"


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


@dataclasses.dataclass
class DbState(State[_State]):
    """
    State that is stored in postgres: DbRunState model.
    """

    id: int
    model: Type[BaseModel]
    type: DbStateRunTypes

    @classmethod
    def new(
        cls, value: _State, *, group_id: int | None = None, type: DbStateRunTypes
    ) -> "DbState[_State]":
        with Session() as session:
            db_state = DbRunState(value=value.model_dump(mode="json"), group_id=group_id, type=type)
            session.add(db_state)
            session.flush()
            value.run_id = db_state.id
            db_state.value = value.model_dump(mode="json")
            session.merge(db_state)
            session.commit()
            return cls(id=db_state.id, model=value.__class__, type=type)

    @classmethod
    def from_id(cls, id: int, model: Type[BaseModel], type: DbStateRunTypes) -> "DbState[_State]":
        return cls(id=id, model=model, type=type)

    def get(self) -> _State:
        with Session() as session:
            db_state = session.get(DbRunState, self.id)

            if db_state is None:
                raise ValueError(f"No state found for id {self.id}")

            if db_state.type != self.type:
                raise ValueError(f"Invalid state type: '{db_state.type}', expected: '{self.type}'")

            return cast(_State, self.model.model_validate(db_state.value))

    def set(self, value: _State):
        with Session() as session:
            db_state = DbRunState(id=self.id, value=value.model_dump(mode="json"))
            session.merge(db_state)
            session.commit()


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
        self.bind.update_state("PROGRESS", meta={"value": value.model_dump(mode="json")})
