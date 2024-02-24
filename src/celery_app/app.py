import logging
import os

import sentry_sdk
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

bootup(__name__, [])
flask_app = Flask(__name__)
flask_app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URL"]

db.init_app(flask_app)

with flask_app.app_context():
    Session.configure(bind=db.engine)


@signals.celeryd_init.connect
def init_sentry(**_kwargs):
    bootup(__name__, [CeleryIntegration(propagate_traces=True)])


@signals.task_failure.connect
def handle_task_failure(**kwargs):
    autofix_logger.error("Task failed", exc_info=kwargs["exception"])
