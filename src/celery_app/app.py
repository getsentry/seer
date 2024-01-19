import os

from celery import Celery

broker_url = os.environ.get("CELERY_BROKER_URL")
if not broker_url:
    raise RuntimeError("CELERY_BROKER_URL must be set")

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
