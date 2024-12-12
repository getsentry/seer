import time
from functools import wraps
from typing import Iterator, List, TypeVar, Any
from sqlalchemy.exc import OperationalError

T = TypeVar('T')

def chunks(lst: List[T], n: int) -> Iterator[List[T]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def with_retry(max_retries: int = 3, backoff_base: int = 2):
    """Decorator that implements retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = backoff_base ** attempt
                        time.sleep(sleep_time)
                    continue
            raise last_exception
        return wrapper
    return decorator

def safe_commit(session):
    """Safely commit database changes with proper error handling."""
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise