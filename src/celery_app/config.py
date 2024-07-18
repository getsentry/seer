from enum import StrEnum
from typing import Any

from celery import Celery

from seer.bootup import module
from seer.configuration import AppConfig


class CeleryConfig(dict[str, Any]):
    pass


class CeleryQueues(StrEnum):
    DEFAULT = "seer"
    CUDA = "seer-cuda"


@module.provider
def celery_config(app_config: AppConfig) -> CeleryConfig:
    return CeleryConfig(
        broker_url=app_config.CELERY_BROKER_URL,
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


@module.provider
def create_celery_app(config: CeleryConfig) -> Celery:
    app = Celery("seer")
    for k, v in config.items():
        setattr(app.conf, k, v)
    return app
