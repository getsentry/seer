from typing import Any

from langsmith import traceable
from sentry_sdk.ai_analytics import ai_track

from celery_app.app import app as celery_app
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisRequest
from seer.automation.autofix.steps.step import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest


class RootCauseStepRequest(PipelineStepTaskRequest):
    pass


@celery_app.task()
def root_cause_task(request: Any):
    return RootCauseStep(request).invoke()


class RootCauseStep(AutofixPipelineStep):
    """
    This class represents the root cause analysis pipeline in the autofix system. It is responsible for
    analyzing the root cause of issues detected in the codebase and suggesting potential fixes.
    """

    @staticmethod
    def get_task():
        return root_cause_task

    @traceable(name="Root Cause", tags=["autofix:v2"])
    @ai_track(description="Root Cause")
    def _invoke(self):
        self.context.event_manager.send_root_cause_analysis_start()

        if self.context.has_missing_codebase_indexes():
            raise ValueError("Codebase indexes must be created before root cause analysis")

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

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()
