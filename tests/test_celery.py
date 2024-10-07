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
                "seer.automation.autofix.steps.change_describer_step.autofix_change_describer_task",
                "seer.automation.autofix.steps.coding_step.autofix_coding_task",
                "seer.automation.autofix.steps.root_cause_step.root_cause_task",
                "seer.automation.autofix.steps.steps.autofix_parallelized_chain_step_task",
                "seer.automation.autofix.steps.steps.autofix_parallelized_conditional_step_task",
                "seer.automation.autofix.tasks.run_autofix_evaluation_on_item",
                "seer.automation.autofix.tasks.check_and_mark_recent_autofix_runs",
                "seer.automation.codegen.unittest_step.unittest_task",
                "seer.automation.tasks.delete_data_for_ttl",
                "seer.smoke_test.smoke_test",
                "seer.automation.tasks.raise_an_exception",  # TODO remove this once testing in prod is done
            ]
        )

        assert set(k for k in celery_app.conf.beat_schedule.keys()) == set(
            [
                "Check and mark recent autofix runs every hour",
                "Delete old Automation runs for 90 day time-to-live",
                "Intentionally raise an error",  # TODO remove this once testing in prod is done
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

        assert set(k for k in app.conf.beat_schedule.keys()) == set([])


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
                "Intentionally raise an error",  # TODO remove this once testing in prod is done
            ]
        )


def test_celery_app_configuration():
    with Module():
        app = Celery(__name__)
        setup_celery_entrypoint(app)
        assert not app.finalized

        app.finalize()
        celery_app.finalize()
        assert app.conf.task_queues == resolve(CeleryConfig)["task_queues"]
        assert celery_app.conf.task_queues == app.conf.task_queues

        assert app.finalized
