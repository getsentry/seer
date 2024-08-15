from celery import Celery

from celery_app.app import celery_app, setup_celery_entrypoint
from celery_app.config import CeleryConfig
from seer.dependency_injection import resolve


def test_detected_celery_jobs():
    assert set(k for k in celery_app.tasks.keys() if not k.startswith("celery.")) == set(
        [
            "seer.automation.autofix.steps.change_describer_step.autofix_change_describer_task",
            "seer.automation.autofix.steps.coding_step.autofix_coding_task",
            "seer.automation.autofix.steps.root_cause_step.root_cause_task",
            "seer.automation.autofix.steps.steps.autofix_parallelized_chain_step_task",
            "seer.automation.autofix.steps.steps.autofix_parallelized_conditional_step_task",
            "seer.automation.autofix.tasks.run_autofix_evaluation_on_item",
            "seer.automation.autofix.tasks.check_and_mark_recent_autofix_runs",
            "seer.smoke_test.smoke_test",
        ]
    )


def test_celery_app_configuration():
    app = Celery(__name__)
    setup_celery_entrypoint(app)
    assert not app.finalized

    app.finalize()
    celery_app.finalize()
    assert app.conf.task_queues == resolve(CeleryConfig)["task_queues"]
    assert celery_app.conf.task_queues == app.conf.task_queues

    assert app.finalized
