import uuid
from typing import Any

import sentry_sdk
from langfuse.decorators import observe

from celery_app.app import celery_app
from seer.automation.agent.models import Message
from seer.automation.autofix.components.confidence import ConfidenceComponent, ConfidenceRequest
from seer.automation.autofix.components.solution.component import SolutionComponent
from seer.automation.autofix.components.solution.models import SolutionRequest
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import AutofixStatus, CommentThread
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.utils import make_kill_signal


class AutofixSolutionStepRequest(PipelineStepTaskRequest):
    initial_memory: list[Message] = []


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
    acks_late=True,
)
def autofix_solution_task(*args, request: dict[str, Any]):
    AutofixSolutionStep(request).invoke()


class AutofixSolutionStep(AutofixPipelineStep):
    """
    This class represents the solution step in the autofix pipeline. It is responsible for
    planning the solution to the issue based on the root cause analysis.
    """

    name = "AutofixSolutionStep"
    max_retries = 2

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> AutofixSolutionStepRequest:
        return AutofixSolutionStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return autofix_solution_task

    @observe(name="Autofix - Solution Step")
    @sentry_sdk.trace
    def _invoke(self, **kwargs):
        super()._invoke()

        self.logger.info("Executing Autofix - Solution Step")

        self.context.event_manager.send_solution_start()

        if self.request.is_retry:
            self.context.event_manager.add_log("Something broke. Re-analyzing from scratch...")
        elif not self.request.initial_memory:
            self.context.event_manager.add_log("Figuring out a solution...")
        else:
            self.context.event_manager.add_log("Going back to the drawing board...")

        state = self.context.state.get()
        root_cause_and_fix, _ = state.get_selected_root_cause()

        if not root_cause_and_fix:
            raise ValueError("Root cause analysis must be performed before solution")

        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        summary = state.request.issue_summary
        if not summary:
            summary = self.context.get_issue_summary()

        # call solution component
        solution_output = SolutionComponent(self.context).invoke(
            SolutionRequest(
                root_cause_and_fix=root_cause_and_fix,
                event_details=event_details,
                summary=summary,
                original_instruction=state.request.instruction,
                initial_memory=self.request.initial_memory,
                profile=state.request.profile,
                trace_tree=state.request.trace_tree,
            )
        )

        state = self.context.state.get()
        if state.steps and state.steps[-1].status == AutofixStatus.WAITING_FOR_USER_RESPONSE:
            return
        if make_kill_signal() in state.signals:
            return

        # send solution result
        self.context.event_manager.send_solution_result(solution_output)
        self.context.event_manager.add_log("Here is Autofix's proposed solution.")

        # confidence evaluation
        if not self.context.state.get().request.options.disable_interactivity:
            run_memory = self.context.get_memory("solution")
            confidence_output = ConfidenceComponent(self.context).invoke(
                ConfidenceRequest(
                    run_memory=run_memory,
                    step_goal_description="figuring out a solution",
                    next_step_goal_description="implementing the solution and drafting a PR",
                )
            )
            if confidence_output:
                with self.context.state.update() as cur:
                    cur.steps[-1].output_confidence_score = (
                        confidence_output.output_confidence_score
                    )
                    cur.steps[-1].proceed_confidence_score = (
                        confidence_output.proceed_confidence_score
                    )
                    sentry_sdk.set_tags(
                        {
                            "has_agent_comment": bool(confidence_output.question),
                        }
                    )
                    if confidence_output.question:
                        cur.steps[-1].agent_comment_thread = CommentThread(
                            id=str(uuid.uuid4()),
                            messages=[
                                Message(role="assistant", content=confidence_output.question)
                            ],
                        )
