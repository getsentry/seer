import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

broker_url = os.environ.get("CELERY_BROKER_URL")
if not broker_url:
    logger.warning("CELERY_BROKER_URL not set")

app = Celery("seer", broker=broker_url)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.enable_utc = True
app.conf.task_default_queue = "seer"
app.conf.task_queues = {
    "seer": {
        "exchange": "seer",
        "routing_key": "seer",
    },
}
app.autodiscover_tasks(["celery_app.tasks"])
