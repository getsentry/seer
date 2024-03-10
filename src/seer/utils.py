import contextlib
import functools
import json
import sys
import weakref
from enum import Enum
from queue import Empty, Full, Queue
from typing import Sequence

from sqlalchemy.orm import DeclarativeBase, Session


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
