import logging
import os

from celery import Celery, signals
from flask import Flask
from sentry_sdk.integrations.celery import CeleryIntegration

from seer.bootup import bootup
from seer.db import Session, db

logger = logging.getLogger(__name__)

broker_url = os.environ.get("CELERY_BROKER_URL")
if not broker_url:
    logger.warning("CELERY_BROKER_URL not set")

autofix_logger = logging.getLogger("autofix")
autofix_logger.setLevel(logging.DEBUG)  # log level debug only for the autofix logger

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

flask_app = bootup(__name__, [CeleryIntegration(propagate_traces=True)])

with flask_app.app_context():
    Session.configure(bind=db.engine)


@signals.task_failure.connect
def handle_task_failure(**kwargs):
    autofix_logger.error("Task failed", exc_info=kwargs["exception"])
