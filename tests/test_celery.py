from multiprocessing import Process

from celery_app.app import celery_app
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
        ]
    )


def test_celery_app_configuration():
    assert not celery_app.finalized

    def _test_celery_app_configuration():
        celery_app.finalize()
        assert celery_app.conf.task_queues == resolve(CeleryConfig)["task_queues"]

    p = Process(target=_test_celery_app_configuration)
    p.start()
    p.join()
    assert (
        p.exitcode == 0
    ), "celery app initialization test failed in separate process, check logs above"

    assert not celery_app.finalized
