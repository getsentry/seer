import logging
import os
from enum import StrEnum
from typing import Annotated

import billiard  # type: ignore[import-untyped]
from celery import Celery, signals
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.celery import CeleryIntegration

from celery_app.app import app as celery_app
from seer.env import Environment
from seer.injector import Dependencies, Injector, Labeled
from seer.logging import LoggingPrefixes, LogLevel

logger = logging.getLogger(__name__)


class CeleryQueues(StrEnum):
    DEFAULT = "seer"
    CUDA = "seer-cuda"


CeleryConfig = Annotated[dict, Labeled("celery_config")]

injector = Injector()

injector.constant(LogLevel, logging.DEBUG)
injector.constant(Celery, celery_app)


@injector.provider
def get_celery_config(env: Environment) -> CeleryConfig:
    return dict(
        broker_url=env.CELERY_BROKER_URL,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        enable_utc=True,
        task_default_queue=CeleryQueues.DEFAULT,
        task_queues={
            CeleryQueues.DEFAULT: {
                "exchange": CeleryQueues.DEFAULT,
                "routing_key": CeleryQueues.DEFAULT,
            },
            CeleryQueues.CUDA: {
                "exchange": CeleryQueues.CUDA,
                "routing_key": CeleryQueues.CUDA,
            },
        },
        result_backend="rpc://",
    )


@injector.initializer
def initialize_celery_app(app: Celery, config: CeleryConfig):
    for k, v in config.items():
        setattr(app.conf, k, v)

    app.autodiscover_tasks(["celery_app.tasks"])

    @signals.celeryd_after_setup.connect
    def capture_worker_name(sender, instance, **kwargs):
        os.environ["WORKER_NAME"] = "{0}".format(sender)

    @signals.task_prerun.connect
    def handle_task_prerun(**kwargs):
        logger.info(
            f"Task started, worker: {os.environ.get('WORKER_NAME')}, process: {billiard.process.current_process().index}"
        )

    @signals.task_failure.connect
    def handle_task_failure(**kwargs):
        logger.error("Task failed", exc_info=kwargs["exception"])

    try:
        yield
    finally:
        signals.celeryd_after_setup.disconnect(capture_worker_name),
        signals.task_prerun.disconnect(handle_task_prerun),
        signals.task_failure.disconnect(handle_task_failure),


@injector.extension
def celery_sentry_integrations() -> list[Integration]:
    return [CeleryIntegration(propagate_traces=True)]


@injector.extension
def logging_prefix() -> LoggingPrefixes:
    return ["celery_app."]


@injector.extension
def celery_dependencies() -> Dependencies:
    from seer import db, env, logging, sentry_config

    return [
        db.injector,
        logging.injector,
        sentry_config.injector,
        env.injector,
    ]
