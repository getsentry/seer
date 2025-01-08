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


def setup_celery_entrypoint(app: Celery):
    """
    Connect all necessary signals for Celery lifecycle management.
    Order of execution:
    1. on_configure - For Celery config only
    2. celeryd_init - For one-time app initialization
    3. worker_ready - For setting up periodic tasks
    """
    app.on_configure.connect(configure_celery)
    signals.celeryd_init.connect(initialize_app)
    signals.worker_ready.connect(setup_tasks)


@inject
def configure_celery(*args: Any, sender: Celery, config: CeleryConfig = injected, **kwargs: Any):
    """Configure Celery-specific settings only"""
    for k, v in config.items():
        setattr(sender.conf, k, v)


@inject
def initialize_app(*args: Any, **kwargs: Any):
    """One-time initialization of the application"""
    bootup(start_model_loading=False, integrations=[CeleryIntegration(propagate_traces=True)])


@inject
def setup_tasks(sender: Celery, **kwargs: Any):
    """Set up periodic tasks after worker is ready"""
    from celery_app.tasks import setup_periodic_tasks
    setup_periodic_tasks(sender)


setup_celery_entrypoint(celery_app)


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
