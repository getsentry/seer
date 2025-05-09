import logging
import os
from typing import Any

import billiard  # type: ignore[import-untyped]
from celery import Celery, signals
from sentry_sdk.integrations.celery import CeleryIntegration

from celery_app.config import CeleryConfig
from seer.bootup import bootup
from seer.dependency_injection import inject, injected
from seer.logging import setup_logger

logger = logging.getLogger(__name__)
celery_app = Celery("seer")


# This abstract helps tests that want to validate the entry point process.
def setup_celery_entrypoint(app: Celery):
    app.on_configure.connect(on_configure)
    app.on_after_finalize.connect(on_after_finalize)


# on_configure signal sent when celery app is being configured. This is *also* called when the celery app is imported and a task is scheduled from outside celery as well.
@inject
def on_configure(*args: Any, sender: Celery, config: CeleryConfig = injected, **kwargs: Any):
    for k, v in config.items():
        setattr(sender.conf, k, v)


# on_after_finalize signal sent after celery app has been finalized. This should only be called from the celery worker and celery beat itself and not from flask or other external sources.
def on_after_finalize(sender, **kwargs):
    from celery_app.tasks import setup_periodic_tasks

    setup_periodic_tasks(sender)


setup_celery_entrypoint(celery_app)


@signals.celeryd_after_setup.connect
def capture_worker_name(sender, instance, **kwargs):
    os.environ["WORKER_NAME"] = "{0}".format(sender)


# celeryd_init signal sent after celery app has been initialized.
@signals.celeryd_init.connect
def bootup_celery_worker(sender, **kwargs):
    bootup(
        start_model_loading=False,
        integrations=[CeleryIntegration(propagate_traces=True)],
    )


# celerybeat_init signal sent after celerybeat app has been initialized.
@signals.beat_init.connect
def bootup_celery_beat(sender, **kwargs):
    bootup(
        start_model_loading=False,
        integrations=[CeleryIntegration(propagate_traces=True)],
    )


@signals.after_task_publish.connect
def handle_task_publish(sender, **kwargs):
    routing_key = kwargs.get("routing_key")
    exchange = kwargs.get("exchange")

    logger.info(f"Task published, task: {sender}, routing_key: {routing_key}, exchange: {exchange}")


@signals.task_prerun.connect
def handle_task_prerun(**kwargs):
    logger.info(
        f"Task started, worker: {os.environ.get('WORKER_NAME')}, process: {billiard.process.current_process().index}"
    )


# For some reason, any celery stubs library does not have this signal when it's actually present...
@signals.task_internal_error.connect  # type: ignore
def handle_task_internal_error(**kwargs):
    logger.error("Task internal error", exc_info=kwargs["exception"])


@signals.task_failure.connect
def handle_task_failure(**kwargs):
    logger.error("Task failed", exc_info=kwargs["exception"])


# 3) Patch Celery's worker logger
@signals.after_setup_logger.connect
def patch_worker_logger(logger: logging.Logger, **kwargs):
    setup_logger(logger)


# 4) Patch Celery's per-task logger (if you want to catch task-level logs too)
@signals.after_setup_task_logger.connect
def patch_task_logger(logger: logging.Logger, **kwargs):
    setup_logger(logger)
