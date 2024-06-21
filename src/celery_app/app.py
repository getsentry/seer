import logging
import os

import billiard  # type: ignore[import-untyped]
from celery import signals
from flask import Flask
from sentry_sdk.integrations.celery import CeleryIntegration

from seer.bootup import bootup, bootup_celery

autofix_logger = logging.getLogger("autofix")
autofix_logger.setLevel(logging.DEBUG)  # log level debug only for the autofix logger

logging.getLogger("automation").setLevel(logging.DEBUG)


app = bootup_celery()
app.autodiscover_tasks(["celery_app.tasks"])

flask_app = Flask(__name__)
flask_app = bootup(flask_app, [CeleryIntegration(propagate_traces=True)])


@signals.celeryd_after_setup.connect
def capture_worker_name(sender, instance, **kwargs):
    os.environ["WORKER_NAME"] = "{0}".format(sender)


@signals.task_prerun.connect
def handle_task_prerun(**kwargs):
    autofix_logger.info(
        f"Task started, worker: {os.environ.get('WORKER_NAME')}, process: {billiard.process.current_process().index}"
    )


@signals.task_failure.connect
def handle_task_failure(**kwargs):
    autofix_logger.error("Task failed", exc_info=kwargs["exception"])
