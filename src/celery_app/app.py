import logging
import os
from typing import Any

import billiard  # type: ignore[import-untyped]
from celery import Celery, signals
from sentry_sdk.integrations.celery import CeleryIntegration

from celery_app.config import CeleryConfig
from seer.bootup import bootup
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)
celery_app = Celery("seer")


@signals.celeryd_init.connect
@inject
def init_celery_app(*args: Any, config: CeleryConfig = injected, **kwargs: Any):
    for k, v in config.items():
        setattr(celery_app.conf, k, v)
    bootup(start_model_loading=False, integrations=[CeleryIntegration(propagate_traces=True)])


@signals.celeryd_after_setup.connect
def capture_worker_name(sender, instance, **kwargs):
    os.environ["WORKER_NAME"] = "{0}".format(sender)


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
