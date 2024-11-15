from enum import StrEnum
from typing import Any

from seer.bootup import module
from seer.configuration import AppConfig
from seer.dependency_injection import injected


class CeleryConfig(dict[str, Any]):
    pass


class CeleryQueues(StrEnum):
    DEFAULT = "seer"
    CUDA = "seer-cuda"


@module.provider
def celery_config(app_config: AppConfig = injected) -> CeleryConfig:
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
            }
        },
        result_backend="rpc://",
    )
