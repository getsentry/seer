# All tasks are part of the import dependency tree of starting from seer endpoints.
# See the test_seer:test_detected_celery_jobs test
from celery.schedules import crontab

import seer.app  # noqa: F401
from celery_app.app import celery_app as celery  # noqa: F401
from seer.anomaly_detection.tasks import (  # noqa: F401
    cleanup_disabled_alerts,
    cleanup_old_timeseries_and_prophet_history,
    cleanup_timeseries_and_predict,
)
from seer.automation.autofix.tasks import check_and_mark_recent_autofix_runs
from seer.automation.tasks import delete_data_for_ttl
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


@inject
def setup_periodic_tasks(sender, config: AppConfig = injected, **kwargs):
    if config.is_autofix_enabled:
        sender.add_periodic_task(
            crontab(minute="0", hour="*"),
            check_and_mark_recent_autofix_runs.signature(
                kwargs={}, queue=config.CELERY_WORKER_QUEUE
            ),
            name="Check and mark recent autofix runs every hour",
        )

        sender.add_periodic_task(
            crontab(minute="0", hour="0"),  # run once a day
            delete_data_for_ttl.signature(kwargs={}, queue=config.CELERY_WORKER_QUEUE),
            name="Delete old Automation runs for 30 day time-to-live",
        )

    # if config.is_autofix_backfill_enabled:
    #     sender.add_periodic_task(
    #         crontab(minute="*/30", hour="*"),  # run once every 30 minutes
    #         collect_all_repos_for_backfill.signature(kwargs={}, queue=config.CELERY_WORKER_QUEUE),
    #         name="Collect all repos for backfill every 30 minutes",
    #     )

    #     sender.add_periodic_task(
    #         crontab(
    #             minute="15,45", hour="*"
    #         ),  # run once every 30 minutes, on a :15 & :45 min to not interfere with autofix backfill
    #         run_repo_sync.signature(kwargs={}, queue=config.CELERY_WORKER_QUEUE),
    #         name="Run repo sync every 30 minutes, on a :15 & :45 min to not interfere with autofix backfill",
    #     )

    #     sender.add_periodic_task(
    #         crontab(minute="0", hour="0"),  # run once a day
    #         run_repo_archive_cleanup.signature(kwargs={}, queue=config.CELERY_WORKER_QUEUE),
    #         name="Run repo archive cleanup every day",
    #     )

    if config.GRPC_SERVER_ENABLE:
        from seer.grpc import try_grpc_client

        sender.add_periodic_task(
            crontab(minute="*", hour="*"),  # run every minute
            try_grpc_client.signature(kwargs={}, queue=config.CELERY_WORKER_QUEUE),
            name="Try executing grpc request every minute.",
        )

    if config.ANOMALY_DETECTION_ENABLED:
        sender.add_periodic_task(
            crontab(minute="0", hour="0", day_of_week="0"),  # Run once a week on Sunday
            cleanup_disabled_alerts.signature(kwargs={}, queue=config.CELERY_WORKER_QUEUE),
            name="Clean up old disabled timeseries every week",
        )

        sender.add_periodic_task(
            crontab(minute="0", hour="0", day_of_week="0"),  # Run once a week on Sunday
            cleanup_old_timeseries_and_prophet_history.signature(
                kwargs={}, queue=config.CELERY_WORKER_QUEUE
            ),
            name="Clean up old timeseries and prophet history every week",
        )
