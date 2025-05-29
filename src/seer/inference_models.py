import contextlib
import functools
import logging
import os
import threading
from typing import Any, Callable, TypeVar

import numpy as np
import sentry_sdk

from seer.anomaly_detection.anomaly_detection import AnomalyDetection
from seer.automation.autofixability import AutofixabilityModel
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected
from seer.grouping.grouping import GroupingLookup
from seer.loading import LoadingResult
from seer.severity.severity_inference import SeverityInference, SeverityRequest

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
    return GroupingLookup(model_path=model_path("issue_grouping_v0/embeddings"))


@deferred_loading("AUTOFIXABILITY_SCORING_ENABLED")
def autofixability_model() -> AutofixabilityModel:
    return AutofixabilityModel(model_path("autofixability_v4/embeddings"))


@deferred_loading("ANOMALY_DETECTION_ENABLED")
@sentry_sdk.trace
def load_anomaly_detection() -> AnomalyDetection:
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
            loaded_models = []
            for item in _deferred:
                model_instance = item()
                loaded_models.append(model_instance)

            # Warm up models that support it
            for model in loaded_models:
                if hasattr(model, "warm_up") and callable(getattr(model, "warm_up")):
                    try:
                        model.warm_up()
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.exception(f"Error warming up model {model.__class__.__name__}: {e}")
                        raise

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


def test_grouping_model() -> bool:
    """Test if the grouping model is working properly.

    Returns:
        bool: True if the model is working, False otherwise.
    """
    try:
        test_text = "Test message for grouping model"
        embedding = grouping_lookup().encode_text(test_text)
        if embedding is None or len(embedding) == 0:
            raise ValueError("Grouping model returned empty embedding")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Grouping model returned non-numeric values in the embedding")

        logging.getLogger(__name__).info("Grouping model test call successful")
        return True
    except Exception as e:
        logging.getLogger(__name__).exception("Grouping model test call failed")
        sentry_sdk.capture_exception(e)
        return False


def test_severity_model() -> bool:
    """Test if the severity model is working properly.

    Returns:
        bool: True if the model is working, False otherwise.
    """
    try:
        test_request = SeverityRequest(message="Test message for severity model")
        embeddings_model().severity_score(test_request)
        logging.getLogger(__name__).info("Severity model test call successful")
        return True
    except Exception as e:
        logging.getLogger(__name__).exception("Severity model test call failed")
        sentry_sdk.capture_exception(e)
        return False
