import contextlib
import functools
import json
import logging
import random
import time
import weakref
from enum import Enum
from queue import Empty, Full, Queue
from typing import Any, Callable, Iterable, Sequence, TypeVar

from sqlalchemy.orm import DeclarativeBase, Session
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def class_method_lru_cache(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


class SeerJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def json_dumps(data, **kwargs) -> str:
    return json.dumps(data, cls=SeerJSONEncoder, **kwargs)


def batch_save_to_db(session: Session, data: Sequence[DeclarativeBase], batch_size: int = 512):
    """
    Save a list of data to the database in batches. Flushes the session after each batch.
    NOTE: Needs to be called inside a session/transaction.
    """
    for i in range(0, len(data), batch_size):
        session.bulk_save_objects(data[i : i + batch_size])

        # Flush to move the data to the db transaction buffer
        session.flush()


@contextlib.contextmanager
def closing_queue(*queues: Queue):
    try:
        yield
    finally:
        for queue in queues:
            try:
                queue.put_nowait(None)
            except Full:
                pass

            try:
                queue.get_nowait()
            except Empty:
                pass


def exception_formatter(exception: Exception) -> str:
    return f"{type(exception).__module__}.{type(exception).__qualname__}: {exception}"


class MaxTriesExceeded(Exception):
    pass


def backoff_on_exception(
    is_exception_retryable: Callable[[Exception], bool],
    max_tries: int = 2,
    sleep_sec_scaler: Callable[[int], float] | None = None,
    jitterer: Callable[[], float] = lambda: random.uniform(0, 0.5),
    long_wait_threshold_sec: float | None = None,
    long_wait_callback: Callable[[], None] | None = None,
):
    """
    Returns a decorator which retries a function on exception iff `is_exception_retryable(exception)`.
    Defaults to exponential backoff with random jitter and one retry.
    """

    if max_tries < 1:
        raise ValueError("max_tries must be at least 1")  # pragma: no cover

    if sleep_sec_scaler is None:
        sleep_sec_scaler = lambda num_tries: min(2**num_tries, 10.0)

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            last_exception = None
            for num_tries in range(1, max_tries + 1):
                try:
                    result = func(*args, **kwargs)
                except Exception as exception:
                    last_exception = exception
                    if is_exception_retryable(exception):
                        sleep_sec = sleep_sec_scaler(num_tries) + jitterer()
                        if (
                            long_wait_threshold_sec is not None
                            and long_wait_callback is not None
                            and sleep_sec > long_wait_threshold_sec
                        ):
                            long_wait_callback()
                        logger.info(
                            f"Encountered {exception_formatter(exception)}. Sleeping for "
                            f"{sleep_sec} seconds before try {num_tries + 1}/{max_tries}."
                        )
                        time.sleep(sleep_sec)
                    else:
                        raise exception
                else:
                    if num_tries > 1:
                        logger.info(f"Retried call successful after {num_tries} tries.")
                    return result

            raise MaxTriesExceeded(
                f"Max tries ({max_tries}) exceeded. "
                f"Last exception: {exception_formatter(last_exception)}"
            ) from last_exception

        return wrapped_func

    return decorator


def retry_once_with_modified_input(
    in_input_bad: Callable[[Exception], bool],
    modify_input_func: Callable[[Any], Any],
    target_kwarg_name: str,
):
    """
    Returns a decorator that retries a function once if `is_input_too_long_func(exception)` is true.
    On retry, it calls `truncate_func` on the value of the keyword argument specified by
    `target_kwarg_name` and calls the decorated function again with the truncated value.

    Requires the target argument to be passed as a keyword argument.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if in_input_bad(e):
                    logger.warning(
                        f"Encountered {exception_formatter(e)}. Attempting to modify input and retry."
                    )

                    if target_kwarg_name not in kwargs:
                        raise ValueError(
                            f"Argument '{target_kwarg_name}' must be passed as a keyword argument "
                            f"to use retry_once_with_modified_input for {func.__name__}."
                        )

                    original_value = kwargs[target_kwarg_name]
                    try:
                        modified_value = modify_input_func(original_value)
                    except Exception as modify_err:
                        logger.error(f"Modification function failed: {modify_err}", exc_info=True)
                        raise e from modify_err

                    new_kwargs = kwargs.copy()
                    new_kwargs[target_kwarg_name] = modified_value

                    logger.info(f"Retrying {func.__name__} with modified '{target_kwarg_name}'.")
                    return func(*args, **new_kwargs)
                else:
                    raise e

        return wrapped_func

    return decorator


def prefix_logger(prefix: str, logger: logging.Logger) -> logging.LoggerAdapter:
    """
    Returns a logger that prefixes all messages with `prefix`.
    """

    class PrefixedLoggingAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f"{prefix}{msg}", kwargs

    return PrefixedLoggingAdapter(logger)


_Batch = TypeVar("_Batch")


def tqdm_sized(
    iterable: Iterable[_Batch],
    length: Callable[[_Batch], int] = len,  # type: ignore[assignment]
    **kwargs,
):
    with tqdm(**kwargs) as pbar:
        for batch in iterable:
            pbar.update(length(batch))
            yield batch
