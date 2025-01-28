import contextlib
import functools
import json
import logging
import random
import time
import weakref
from enum import Enum
from queue import Empty, Full, Queue
from typing import Callable, Sequence

from sqlalchemy.orm import DeclarativeBase, Session

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


def backoff_on_exception(
    is_exception_retryable: Callable[[Exception], bool],
    max_tries: int = 2,
    sleep_sec_scaler: Callable[[int], float] | None = None,
    jitterer: Callable[[], float] = lambda: random.uniform(0, 0.5),
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
            num_tries = 0
            last_exception = None
            while num_tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as exception:
                    num_tries += 1
                    last_exception = exception
                    if is_exception_retryable(exception):
                        sleep_sec = sleep_sec_scaler(num_tries) + jitterer()
                        logger.info(
                            f"Encountered {type(exception).__name__}: {exception}. Sleeping for "
                            f"{sleep_sec} seconds before attempting retry {num_tries}/{max_tries}."
                        )
                        time.sleep(sleep_sec)
                    else:
                        raise exception
            raise last_exception

        return wrapped_func

    return decorator
