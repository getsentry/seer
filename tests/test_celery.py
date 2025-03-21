from celery import Celery

import celery_app.tasks
from celery_app.app import celery_app, setup_celery_entrypoint
from celery_app.config import CeleryConfig
from seer.configuration import AppConfig
from seer.dependency_injection import Module, resolve


# Validates the total sum of all celery jobs with the broadest configuration.
def test_detected_celery_jobs():
    with Module():
        assert set(k for k in celery_app.tasks.keys() if not k.startswith("celery.")) == set(
            [
                "seer.anomaly_detection.tasks.cleanup_timeseries_and_predict",
                "seer.anomaly_detection.tasks.cleanup_old_timeseries_and_prophet_history",
                "seer.anomaly_detection.tasks.cleanup_disabled_alerts",
                "seer.automation.autofix.steps.change_describer_step.autofix_change_describer_task",
                "seer.automation.autofix.steps.coding_step.autofix_coding_task",
                "seer.automation.autofix.steps.root_cause_step.root_cause_task",
                "seer.automation.autofix.steps.solution_step.autofix_solution_task",
                "seer.automation.autofix.steps.steps.autofix_parallelized_chain_step_task",
                "seer.automation.autofix.steps.steps.autofix_parallelized_conditional_step_task",
                "seer.automation.autofix.tasks.check_and_mark_recent_autofix_runs",
                "seer.automation.autofix.tasks.commit_changes_task",
                "seer.automation.autofix.tasks.run_autofix_evaluation_on_item",
                "seer.automation.codegen.unittest_step.unittest_task",
                "seer.automation.codegen.pr_review_step.pr_review_task",
                "seer.automation.codegen.relevant_warnings_step.relevant_warnings_task",
                "seer.automation.codegen.retry_unittest_step.retry_unittest_task",
                "seer.automation.tasks.delete_data_for_ttl",
                "seer.smoke_test.smoke_test",
            ]
        )

        assert set(k for k in celery_app.conf.beat_schedule.keys()) == set(
            [
                "Check and mark recent autofix runs every hour",
                "Delete old Automation runs for 90 day time-to-live",
                "Clean up old disabled timeseries every week",
                "Clean up old timeseries and prophet history every week",
            ]
        )


def test_anomaly_beat_jobs():
    with Module():
        app = Celery(__name__)
        setup_celery_entrypoint(app)
        app_config = resolve(AppConfig)
        app_config.disable_all()
        app_config.ANOMALY_DETECTION_ENABLED = True
        app.finalize()

        assert set(k for k in app.conf.beat_schedule.keys()) == set(
            [
                "Clean up old disabled timeseries every week",
                "Clean up old timeseries and prophet history every week",
            ]
        )


def test_autofix_beat_jobs():
    with Module():
        app = Celery(__name__)
        setup_celery_entrypoint(app)
        app_config = resolve(AppConfig)
        app_config.disable_all()
        app_config.AUTOFIX_ENABLED = True
        app.finalize()

        assert set(k for k in app.conf.beat_schedule.keys()) == set(
            [
                "Check and mark recent autofix runs every hour",
                "Delete old Automation runs for 90 day time-to-live",
            ]
        )


def test_celery_app_configuration():
    with Module():
        app = Celery(__name__)
        setup_celery_entrypoint(app)
        assert not app.finalized

        app.finalize()
        celery_app.finalize()

        assert app.conf.task_default_queue == resolve(CeleryConfig)["task_default_queue"]
        assert app.conf.task_queues == resolve(CeleryConfig)["task_queues"]
        assert celery_app.conf.task_queues == app.conf.task_queues

        assert app.finalized
