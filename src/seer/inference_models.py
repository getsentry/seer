import contextlib
import functools
import os
import threading
from typing import Any, Callable, TypeVar

import sentry_sdk

from seer.anomaly_detection.anomaly_detection import AnomalyDetection
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected
from seer.grouping.grouping import GroupingLookup
from seer.loading import LoadingResult
from seer.severity.severity_inference import SeverityInference

root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))


def model_path(subpath: str) -> str:
    return os.path.join(root, "models", subpath)


_A = TypeVar("_A")
_deferred: list[Callable[[], Any]] = []


def deferred_loading(env_var_required: str) -> Callable[[Callable[[], _A]], Callable[[], _A]]:
    def decorator(func: Callable[[], _A]) -> Callable[[], Any]:
        if os.environ.get(env_var_required, "").lower() not in ("true", "t", "1"):
            return func
        # python functools is thread safe: https://docs.python.org/3/library/functools.html
        wrapped = functools.cache(func)
        _deferred.append(wrapped)
        return wrapped

    return decorator


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


@deferred_loading("ANOMALY_DETECTION_ENABLED")
def anomaly_detection() -> AnomalyDetection:
    return AnomalyDetection()


_loading_lock: threading.RLock = threading.RLock()
_loading_thread: threading.Thread | None = None
_loading_result: LoadingResult = LoadingResult.PENDING


def models_loading_status() -> LoadingResult:
    global _loading_result
    return _loading_result


def start_loading() -> threading.Thread:
    global _loading_result

    with _loading_lock:
        if _loading_result != LoadingResult.PENDING:
            raise RuntimeError(
                "start_loading invoked, but loading already started.  call reset_loading_state"
            )
        _loading_result = LoadingResult.LOADING

    def load():
        global _loading_result
        try:
            for item in _deferred:
                item()
            with _loading_lock:
                _loading_result = LoadingResult.DONE
        except Exception:
            sentry_sdk.capture_exception()
            with _loading_lock:
                _loading_result = LoadingResult.FAILED

    thread = threading.Thread(target=load)
    thread.start()
    return thread


def reset_loading_state(state: LoadingResult = LoadingResult.PENDING):
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


@inject
def initialize_models(start_model_loading: bool, config: AppConfig = injected):
    if start_model_loading:
        start_loading()
    torch_num_threads = config.TORCH_NUM_THREADS
    if torch_num_threads:
        import torch

        torch.set_num_threads(torch_num_threads)
