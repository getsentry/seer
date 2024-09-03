from typing import Any

from langfuse.decorators import observe
from pydantic import ValidationError
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
from seer.automation.summarize.issue import IssueSummary
from seer.db import DbIssueSummary, Session


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

        state = self.context.state.get()
        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        group_id = state.request.issue.id
        summary = None
        with Session() as session:
            group_summary = session.get(DbIssueSummary, group_id)
            if group_summary:
                try:
                    summary = IssueSummary.model_validate(group_summary.summary)
                except ValidationError:
                    pass

        root_cause_output = RootCauseAnalysisComponent(self.context).invoke(
            RootCauseAnalysisRequest(
                event_details=event_details, instruction=state.request.instruction, summary=summary
            )
        )

        self.context.event_manager.send_root_cause_analysis_result(root_cause_output)
