import contextlib
import functools
import os
import threading
from typing import Annotated, Any, Callable, Literal, Protocol, TypeVar

import sentry_sdk

from seer.env import Environment
from seer.grouping.grouping import GroupingLookup
from seer.injector import Dependencies, Injector, Labeled
from seer.severity.severity_inference import SeverityInference

root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

injector = Injector()

AsyncLoad = Annotated[bool, Labeled("async_model_load")]

injector.constant(AsyncLoad, False)


@injector.extension
def dependencies() -> Dependencies:
    from seer import env

    return [env.injector]


@injector.initializer
def initialize_models(async_load: AsyncLoad, env: Environment):
    import torch

    torch.set_num_threads(env.TORCH_NUM_THREADS)

    if async_load:
        start_loading(async_load)
        try:
            yield
        finally:
            reset_loading_state()


def model_path(subpath: str) -> str:
    return os.path.join(root, "models", subpath)


_A = TypeVar("_A")


class DeferredLoading(Protocol):
    def __call__(self) -> Any: ...


_deferred: list[DeferredLoading] = []


def deferred_loading(env_var_required: str) -> Callable[[Callable[[], _A]], Callable[[], _A]]:
    def decorator(func: Callable[[], _A]) -> Callable[[], Any]:
        if os.environ.get(env_var_required, "").lower() not in ("true", "t", "1"):
            return func
        # python functools is thread safe: https://docs.python.org/3/library/functools.html
        wrapped = functools.cache(func)
        _deferred.append(wrapped)
        return wrapped

    return decorator


@injector.extension
def default_deferred_loading() -> list[DeferredLoading]:
    return _deferred


@deferred_loading("SEVERITY_ENABLED")
def embeddings_model() -> SeverityInference:
    return SeverityInference(
        model_path("issue_severity_v0/embeddings"), model_path("issue_severity_v0/classifier")
    )


@deferred_loading("GROUPING_ENABLED")
def grouping_lookup() -> GroupingLookup:
    return GroupingLookup(
        model_path=model_path("issue_grouping_v0/embeddings"),
        data_path=model_path("issue_grouping_v0/data.pkl"),
    )


LoadingResult = Literal["pending", "loading", "done", "failed"]
_loading_lock: threading.RLock = threading.RLock()
_loading_thread: threading.Thread | None = None
_loading_result: LoadingResult = "pending"


def models_loading_status() -> LoadingResult:
    global _loading_result
    return _loading_result


def start_loading(load_async: bool):
    global _loading_result

    with _loading_lock:
        if _loading_result != "pending":
            raise RuntimeError(
                "start_loading invoked, but loading already started.  call reset_loading_state"
            )
        _loading_result = "loading"

    def load():
        global _loading_result
        try:
            for item in _deferred:
                item()
            with _loading_lock:
                _loading_result = "done"
        except Exception:
            sentry_sdk.capture_exception()
            with _loading_lock:
                _loading_result = "failed"

    thread = threading.Thread(target=load)
    thread.start()

    if not load_async:
        thread.join(timeout=5)


def reset_loading_state(state: LoadingResult = "pending"):
    global _loading_thread, _deferred, _loading_result

    with _loading_lock:
        if _loading_thread:
            _loading_thread.join()
        _loading_thread = None
        _loading_result = state

    for f in _deferred:
        if hasattr(f, "cache_clear"):
            f.cache_clear()


@contextlib.contextmanager
def dummy_deferred(c: Callable):
    global _deferred
    old = _deferred[:]
    try:
        _deferred = [c]
        yield
    finally:
        _deferred = old
