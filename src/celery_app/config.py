from typing import Any

from seer.bootup import module
from seer.configuration import AppConfig
from seer.dependency_injection import injected


class CeleryConfig(dict[str, Any]):
    pass


@module.provider
def celery_config(app_config: AppConfig = injected) -> CeleryConfig:
    return CeleryConfig(
        broker_url=app_config.CELERY_BROKER_URL,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        enable_utc=True,
        task_default_queue=app_config.CELERY_WORKER_QUEUE,
        task_queues={
            app_config.CELERY_WORKER_QUEUE: {
                "exchange": app_config.CELERY_WORKER_QUEUE,
                "routing_key": app_config.CELERY_WORKER_QUEUE,
            }
        },
        result_backend="rpc://",
        worker_max_tasks_per_child=8,
    )
