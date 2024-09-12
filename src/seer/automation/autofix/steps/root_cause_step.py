from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisRequest
from seer.automation.autofix.config import (
    AUTOFIX_ROOT_CAUSE_HARD_TIME_LIMIT_SECS,
    AUTOFIX_ROOT_CAUSE_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest


class RootCauseStepRequest(PipelineStepTaskRequest):
    pass


@celery_app.task(
    time_limit=AUTOFIX_ROOT_CAUSE_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_ROOT_CAUSE_SOFT_TIME_LIMIT_SECS,
)
def root_cause_task(*args, request: Any):
    return RootCauseStep(request).invoke()


class RootCauseStep(AutofixPipelineStep):
    """
    This class represents the root cause analysis pipeline in the Autofix system. It is responsible for
    analyzing the root cause of issues detected in the codebase.
    """

    name = "RootCauseStep"

    max_retries = 2

    @staticmethod
    def get_task():
        return root_cause_task

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> RootCauseStepRequest:
        return RootCauseStepRequest.model_validate(request)

    @observe(name="Autofix - Root Cause Step")
    @ai_track(description="Autofix - Root Cause Step")
    def _invoke(self, **kwargs):
        self.context.event_manager.send_root_cause_analysis_start()

        self.context.event_manager.add_log("Beginning root cause analysis...")

        state = self.context.state.get()
        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        summary = state.request.issue_summary
        if not summary:
            summary = self.context.get_issue_summary()

        root_cause_output = RootCauseAnalysisComponent(self.context).invoke(
            RootCauseAnalysisRequest(
                event_details=event_details, instruction=state.request.instruction, summary=summary
            )
        )

        self.context.event_manager.send_root_cause_analysis_result(root_cause_output)
        self.context.event_manager.add_log("Here's what I think the root cause is. If you disagree, feel free to edit it or provide your own idea below.")
