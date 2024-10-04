from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from celery_app.config import CeleryQueues
from seer.automation.agent.models import Message
from seer.automation.autofix.components.coding.component import CodingComponent
from seer.automation.autofix.components.coding.models import CodingRequest
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import AutofixStatus
from seer.automation.autofix.steps.change_describer_step import (
    AutofixChangeDescriberRequest,
    AutofixChangeDescriberStep,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.utils import make_kill_signal


class AutofixCodingStepRequest(PipelineStepTaskRequest):
    initial_memory: list[Message] = []


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def autofix_coding_task(*args, request: dict[str, Any]):
    AutofixCodingStep(request).invoke()


class AutofixCodingStep(AutofixPipelineStep):
    """
    This class represents the coding step in the autofix pipeline. It is responsible for
    executing the fixes suggested by the coding component based on the root cause analysis.
    """

    name = "AutofixCodingStep"
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> AutofixCodingStepRequest:
        return AutofixCodingStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return autofix_coding_task

    @observe(name="Autofix - Plan+Code Step")
    @ai_track(description="Autofix - Plan+Code Step")
    def _invoke(self, **kwargs):
        self.context.event_manager.clear_file_changes()

        self.logger.info("Executing Autofix - Plan+Code Step")

        self.context.event_manager.send_coding_start()
        if not self.request.initial_memory:
            self.context.event_manager.add_log(
                "Figuring out a fix for the root cause of this issue..."
            )
        else:
            self.context.event_manager.add_log("Continuing to analyze...")

        state = self.context.state.get()
        root_cause_and_fix = state.get_selected_root_cause_and_fix()

        if not root_cause_and_fix:
            raise ValueError("Root cause analysis must be performed before coding")

        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        summary = state.request.issue_summary
        if not summary:
            summary = self.context.get_issue_summary()

        coding_output = CodingComponent(self.context).invoke(
            CodingRequest(
                event_details=event_details,
                root_cause_and_fix=root_cause_and_fix,
                instruction=state.request.instruction,
                summary=summary,
                initial_memory=self.request.initial_memory,
            )
        )

        state = self.context.state.get()
        if state.steps[-1].status == AutofixStatus.WAITING_FOR_USER_RESPONSE:
            return
        if make_kill_signal() in state.signals:
            return

        self.context.event_manager.send_coding_result(coding_output)
  
        should_make_pr_automatically = state.request.options.comment_on_pr_with_url is not None
        self.next(
            AutofixChangeDescriberStep.get_signature(
                AutofixChangeDescriberRequest(
                    **self.step_request_fields,
                    should_make_pr_automatically=should_make_pr_automatically,
                ),
            ),
            queue=CeleryQueues.DEFAULT,
        )
