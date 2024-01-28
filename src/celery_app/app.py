import logging
import os

import sentry_sdk
from celery import Celery, signals
from sentry_sdk.integrations.celery import CeleryIntegration

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


@signals.celeryd_init.connect
def init_sentry(**_kwargs):
    print("init_sentry DONE")
    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        integrations=[CeleryIntegration(propagate_traces=True)],
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        enable_tracing=True,
    )
