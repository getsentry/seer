# All tasks are part of the import dependency tree of starting from seer endpoints.
# See the test_seer:test_detected_celery_jobs test
from celery.schedules import crontab

import seer.app  # noqa: F401
from celery_app.app import celery_app as celery  # noqa: F401
from celery_app.config import CeleryQueues
from seer.automation.autofix.tasks import check_and_mark_recent_autofix_runs
from seer.automation.tasks import delete_data_for_ttl
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


@inject
def setup_periodic_tasks(sender, config: AppConfig = injected, **kwargs):
    if config.is_autofix_enabled:
        sender.add_periodic_task(
            crontab(minute="0", hour="*"),
            check_and_mark_recent_autofix_runs.signature(kwargs={}, queue=CeleryQueues.DEFAULT),
            name="Check and mark recent autofix runs every hour",
        )

        sender.add_periodic_task(
            crontab(minute="0", hour="0"),  # run once a day
            delete_data_for_ttl.signature(kwargs={}, queue=CeleryQueues.DEFAULT),
            name="Delete old Automation runs for 90 day time-to-live",
        )
        # TODO remove this task, it's just for testing in prod; throws an error every minute
        sender.add_periodic_task(
            crontab(minute="*", hour="*"),
            buggy_code.signature(kwargs={}, queue=CeleryQueues.DEFAULT),
            name="Intentionally raise an error",
        )

    if config.GRPC_SERVER_ENABLE:
        from seer.grpc import try_grpc_client

        sender.add_periodic_task(
            crontab(minute="*", hour="*"),  # run every minute
            try_grpc_client.signature(kwargs={}, queue=CeleryQueues.DEFAULT),
            name="Try executing grpc request every minute.",
        )
