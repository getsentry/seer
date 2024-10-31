import contextlib

from celery_app.app import celery_app


@contextlib.contextmanager
def eager_celery():
    orig = celery_app.conf.task_always_eager
    celery_app.conf.task_always_eager = True

    try:
        yield
    finally:
        celery_app.conf.task_always_eager = orig
