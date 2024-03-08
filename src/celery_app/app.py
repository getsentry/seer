import logging

from celery import signals
from sentry_sdk.integrations.celery import CeleryIntegration

from seer.bootup import bootup, bootup_celery

autofix_logger = logging.getLogger("autofix")
autofix_logger.setLevel(logging.DEBUG)  # log level debug only for the autofix logger

app = bootup_celery()
app.autodiscover_tasks(["celery_app.tasks"])

flask_app = bootup(__name__, [CeleryIntegration(propagate_traces=True)])


@signals.task_failure.connect
def handle_task_failure(**kwargs):
    autofix_logger.error("Task failed", exc_info=kwargs["exception"])
