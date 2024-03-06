import abc
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel

_State = TypeVar("_State", bound=BaseModel)


class StateInvalidError(Exception):
    """
    Throw when retrieving a state value, but it did not pass validation, or could not be migrated.

    In this case, the caller should likely decide whether to rebuild the state from a third part source,
    or to fail the process.
    """

    pass


class StateUnavailableError(Exception):
    """
    Throw when retrieving a state value, but it was not possible to retrieve it.  In general, if
    is_available returns True, this likely won't be thrown, but due to concurrency issues, it may be possible
    that an attempt to retrieve state will fail nonetheless.

    In this case, the called should likely decide whether to rebuild the state from a third part source,
    or to fail the process.
    """

    pass


class State(abc.ABC, Generic[_State]):
    """
    Am abstraction of state that considers reconciliation between a source of truth and a potential
    external corroboration of that state.  This important in cases where serialization or durability are
    not strongly guaranteed, and we want to handle cases where state will break down and need to be rebuilt.
    """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """
        Return true iff the state is available from its source of truth.
        """
        pass

    @abc.abstractmethod
    def get_and_validate(self) -> _State:
        """
        Attempts to fetch the stored, local state and validate it, throwing an exception if it was
        not possible to retrieve it.
        """
        pass

    @abc.abstractmethod
    def rebuild(self) -> _State:
        """
        Fetch the state from an alter
        :return:
        """
        pass

    @abc.abstractmethod
    def store_state(self, state: _State) -> bool:
        """
        Attempts to store the state, but may fail in the face of concurrent writes or other failures.
        Returns true iff the state successfully stored.
        """
        pass


class DelegatesRebuildState(State[_State]):
    primary: State[_State]
    delegated_rebuild: State[_State]

    def __init__(self, primary: State[_State], delegated_rebuild: State[_State]):
        self.primary = primary
        self.delegated_rebuild = delegated_rebuild

    def is_available(self) -> bool:
        return self.primary.is_available()

    def get_and_validate(self) -> _State:
        return self.primary.get_and_validate()

    def rebuild(self) -> _State:
        return self.delegated_rebuild.rebuild()

    def store_state(self, state: _State) -> bool:
        return self.primary.store_state(state)


class ReadOnlyState(State[_State]):
    source: Callable[[], _State]

    def __init__(self, source: Callable[[], _State]):
        self.source = source

    def is_available(self) -> bool:
        return True

    def get_and_validate(self) -> _State:
        try:
            return self.source()
        except Exception as e:
            raise StateUnavailableError from e

    def rebuild(self) -> _State:
        return self.source()

    def store_state(self, state: _State) -> bool:
        return False


class InMemoryState(State[_State]):
    state: _State | None = None
    default: _State

    def __init__(self, default: _State) -> None:
        self.default = default

    def is_available(self) -> bool:
        return self.state is not None

    def get_and_validate(self) -> _State:
        if self.state is not None:
            return self.state
        raise StateUnavailableError

    def rebuild(self) -> _State:
        return self.default

    def store_state(self, state: _State) -> bool:
        self.state = state
        return True


class JsonSerializedState(State[_State]):
    state: dict[str, Any] | bytes | None = None
    default: _State

    def __init__(self, default: _State) -> None:
        self.default = default

    def is_available(self) -> bool:
        return self.state is not None

    def get_and_validate(self) -> _State:
        if self.state is not None:
            if isinstance(self.state, bytes):
                return type(self.default).parse_raw(self.state)
            return type(self.default).parse_obj(self.state)
        raise StateUnavailableError

    def rebuild(self) -> _State:
        return self.default

    def store_state(self, state: _State) -> bool:
        self.state = state.model_dump()
        return True
