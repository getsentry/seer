from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import app as celery_app
from celery_app.config import CeleryQueues
from seer.automation.autofix.components.planner.component import PlanningComponent
from seer.automation.autofix.components.planner.models import PlanningRequest
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import AutofixTextInstruction
from seer.automation.autofix.steps.execution_step import (
    AutofixExecutionStep,
    AutofixExecutionStepRequest,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineChain, PipelineStepTaskRequest


class AutofixPlanningStepRequest(PipelineStepTaskRequest):
    instruction: AutofixTextInstruction | None = None


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def autofix_planning_task(*args, request: dict[str, Any]):
    AutofixPlanningStep(request).invoke()


class AutofixPlanningStep(PipelineChain, AutofixPipelineStep):
    """
    This class represents the execution pipeline in the autofix system. It is responsible for
    executing the fixes suggested by the planning component based on the root cause analysis.
    """

    name = "AutofixPlanningChainStep"

    request: AutofixPlanningStepRequest

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> AutofixPlanningStepRequest:
        return AutofixPlanningStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return autofix_planning_task

    @observe(name="Autofix - Planning Step")
    @ai_track(description="Autofix - Planning Step")
    def _invoke(self, **kwargs):
        self.context.event_manager.send_codebase_indexing_complete_if_exists()

        self.context.event_manager.send_planning_start(is_update=bool(self.request.instruction))

        if self.context.has_missing_codebase_indexes():
            raise ValueError("Codebase indexes must be created before planning")

        state = self.context.state.get()
        root_cause_and_fix = state.get_selected_root_cause_and_fix()

        if not root_cause_and_fix:
            raise ValueError("Root cause analysis must be performed before planning")

        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        planning_output = PlanningComponent(self.context).invoke(
            PlanningRequest(
                event_details=event_details,
                root_cause_and_fix=root_cause_and_fix,
                original_instruction=state.request.instruction,
                new_instruction=self.request.instruction.text if self.request.instruction else None,
            )
        )

        self.context.event_manager.send_planning_result(planning_output)

        if not planning_output:
            return

        # Call the first step in the execution chain
        self.next(
            AutofixExecutionStep.get_signature(
                AutofixExecutionStepRequest(
                    **self.step_request_fields,
                    task_index=0,
                    planning_output=planning_output,
                )
            ),
            queue=CeleryQueues.CUDA,
        )
