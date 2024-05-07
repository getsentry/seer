from typing import Any

from celery_app.app import app as celery_app
from seer.automation.autofix.steps.step import AutofixPipelineStep
from seer.automation.autofix.utils import autofix_logger
from seer.automation.pipeline import PipelineStepTaskRequest


class CheckIndexingDoneStepRequest(PipelineStepTaskRequest):
    expected_signal_keys: list[str]
    success_link: Any


@celery_app.task()
def check_indexing_done_task(request: Any, *args):
    CheckIndexingDoneStep(request).invoke()


class CheckIndexingDoneStep(AutofixPipelineStep):
    request_class = CheckIndexingDoneStepRequest

    request: CheckIndexingDoneStepRequest

    @staticmethod
    def get_task():
        return check_indexing_done_task

    def _invoke(self):
        signals = self.context.state.get().signals
        if not all(signal in signals for signal in self.request.expected_signal_keys):
            autofix_logger.debug(f"Codebase indexing not done for all repos")
            return

        autofix_logger.debug(f"Codebase indexing done for all repos")
        self.request.success_link.apply_async()

    def _handle_exception(self, exception: Exception):
        pass
