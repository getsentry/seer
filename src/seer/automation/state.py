import abc
import contextlib
import dataclasses
import functools
import threading
import time  # Add time import
from enum import Enum
from typing import Any, ContextManager, Generic, Iterator, Type, TypeVar

from pydantic import BaseModel
from sqlalchemy import select

from seer.db import DbRunState, Session

_State = TypeVar("_State", bound=BaseModel)
_StateB = TypeVar("_StateB", bound=BaseModel)


class DbStateRunTypes(str, Enum):
    AUTOFIX = "autofix"
    UNIT_TEST = "unit-test"
    PR_REVIEW = "pr-review"
    RELEVANT_WARNINGS = "relevant-warnings"
    PR_CLOSED = "pr-closed"
    UNIT_TESTS_RETRY = "unit-test-retry"


# Create an in-memory cache for state objects
_state_cache = {}
_state_cache_lock = threading.RLock()


def memoize_state_get(ttl_seconds=2):
 """Decorator to cache state.get() calls for a short time to reduce DB queries"""

 def decorator(func):
 @functools.wraps(func)
 def wrapper(self, *args, **kwargs):
 cache_key = (self.__class__.__name__, self.id)

 with _state_cache_lock:
 now = time.time()
 if cache_key in _state_cache:
 cached_result, timestamp = _state_cache[cache_key]
 if now - timestamp < ttl_seconds:
 return cached_result

 result = func(self, *args, **kwargs)

 with _state_cache_lock:
 _state_cache[cache_key] = (result, time.time())

 # Clean old cache entries occasionally
 if len(_state_cache) > 1000: # Arbitrary limit
 now = time.time()
 for k in list(_state_cache.keys()):
 if now - _state_cache[k][1] > ttl_seconds * 2:
 del _state_cache[k]

 return result
 return wrapper
 return decorator


class State(abc.ABC, Generic[_State]):
    """
    An abstract state buffer that attempts to push state changes to a sink.
    """

    @abc.abstractmethod
    def get(self) -> _State:
        """
        A method to return a locally secure copy of the state.  Note that mutations to this
        value are not likely to be reflected, see `update`.
        """
        pass

    @abc.abstractmethod
    def update(self) -> ContextManager[_State]:
        """
        A method to atomically get and mutate a state value.  Subclasses should implement
        concurrency primitives around this method to ensure that concurrent access is limited
        by the context of the update itself.
        :return:
        """
        pass


@dataclasses.dataclass
class LocalMemoryState(State[_State]):
    val: _State
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    def get(self) -> _State:
        return self.val

    @contextlib.contextmanager
    def update(self):
        with self.lock:
            val = self.get()
            yield val
            # Mostly a no-op, except in the case that `get` has semantics copying
            self.val = val


@dataclasses.dataclass
class DbState(State[_State]):
    """
    State that is stored in postgres: DbRunState model.
    """

    id: int
    model: Type[_State]
    type: DbStateRunTypes

    @classmethod
    def new(
        cls, value: _State, *, group_id: int | None = None, t: DbStateRunTypes
    ) -> "DbState[_State]":
        with Session() as session:
            db_state = DbRunState(value=value.model_dump(mode="json"), group_id=group_id, type=t)
            session.add(db_state)
            session.flush()
            value.run_id = db_state.id
            db_state.value = value.model_dump(mode="json")
            session.merge(db_state)
            session.commit()
            return cls(id=db_state.id, model=type(value), type=t)

 @memoize_state_get(ttl_seconds=2)
    def get(self) -> _State:
        with Session() as session:
            db_state = session.get(DbRunState, self.id)
            self.validate(db_state)
            assert db_state
            return self.model.model_validate(db_state.value)

    def validate(self, db_state: DbRunState | None):
        if db_state is None:
            raise ValueError(f"No state found for id {self.id}")
        if db_state.type != self.type:
            raise ValueError(f"Invalid state type: '{db_state.type}', expected: '{self.type}'")

    def apply_to_run_state(self, value: _State, run_state: DbRunState):
        """
        Can be used to pass down context from state into the db context
        """
        pass

    def before_update(self, value: _State):
        """
        Can be used to run some logic before the update is applied to the db
        """
        pass

    @contextlib.contextmanager
    def update(self):
        """
        Uses a 'with for update' clause on the db run id, ensuring it is safe against concurrent transactions.
        Note however, that if you have two competing updates in which neither can fully complete (say a circle
        of inter related locks), the database may reach a deadlock state which last until the lock timeout configured
        on the postgres database.
        """
 # Clear cache for this state before update
 cache_key = (self.__class__.__name__, self.id)
 with _state_cache_lock:
 if cache_key in _state_cache:
 del _state_cache[cache_key]

        with Session() as session:
            r = session.execute(
                select(DbRunState).where(DbRunState.id == self.id).with_for_update()
            ).scalar_one_or_none()
            self.validate(r)
            assert r
            value = self.model.model_validate(r.value)
            yield value
            self.before_update(value)
            db_state = DbRunState(id=self.id, value=value.model_dump(mode="json"))
            self.apply_to_run_state(value, db_state)
            session.merge(db_state)
            session.commit()

 # Update cache with new value after successful commit
 with _state_cache_lock:
 _state_cache[cache_key] = (value, time.time())


@functools.total_ordering
class BufferedMemoryState(State[_State]):
    values: list[_State] = dataclasses.field(default_factory=list)
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    def __init__(self, initial: _State):
        self.values.append(initial)

    def get(self) -> _State:
        return self.values[-1]

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

    @contextlib.contextmanager
    def update(self):
        with self.lock:
            value = self.get()
            yield value
            self.values.append(value)
