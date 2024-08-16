# All tasks are part of the import dependency tree of starting from seer endpoints.
# See the test_seer:test_detected_celery_jobs test
from celery.schedules import crontab

import seer.app  # noqa: F401
from celery_app.app import celery_app as celery  # noqa: F401
from celery_app.config import CeleryQueues
from seer.automation.autofix.tasks import (
    check_and_mark_recent_autofix_runs,
    delete_old_autofix_runs,
)


@celery.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        crontab(minute="0", hour="*"),
        check_and_mark_recent_autofix_runs.signature(kwargs={}, queue=CeleryQueues.DEFAULT),
        name="Check and mark recent autofix runs every hour",
    )

    sender.add_periodic_task(
        crontab(minute="0", hour="0"),  # run once a day
        delete_old_autofix_runs.signature(kwargs={}, queue=CeleryQueues.DEFAULT),
        name="Delete old Autofix runs for 90 day time-to-live",
    )
