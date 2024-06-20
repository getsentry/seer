from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import app as celery_app
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
    This class represents the root cause analysis pipeline in the autofix system. It is responsible for
    analyzing the root cause of issues detected in the codebase and suggesting potential fixes.
    """

    name = "RootCauseStep"

    @staticmethod
    def get_task():
        return root_cause_task

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> RootCauseStepRequest:
        return RootCauseStepRequest.model_validate(request)

    @observe(name="Autofix - Root Cause Step")
    @ai_track(description="Autofix - Root Cause Step")
    def _invoke(self, **kwargs):
        self.context.event_manager.send_codebase_indexing_complete_if_exists()
        self.context.event_manager.send_root_cause_analysis_start()

        if self.context.has_missing_codebase_indexes():
            raise RuntimeError("Codebase indexes must be created before root cause analysis")

        state = self.context.state.get()
        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        root_cause_output = RootCauseAnalysisComponent(self.context).invoke(
            RootCauseAnalysisRequest(
                event_details=event_details,
                instruction=state.request.instruction,
            )
        )

        self.context.event_manager.send_root_cause_analysis_result(root_cause_output)
