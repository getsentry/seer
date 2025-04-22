import uuid
from typing import Any

import sentry_sdk
from langfuse.decorators import observe

from celery_app.app import celery_app
from seer.automation.agent.models import Message
from seer.automation.autofix.components.coding.component import CodingComponent
from seer.automation.autofix.components.coding.models import CodingRequest
from seer.automation.autofix.components.confidence import ConfidenceComponent, ConfidenceRequest
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import AutofixStatus, CommentThread
from seer.automation.autofix.steps.change_describer_step import (
    AutofixChangeDescriberRequest,
    AutofixChangeDescriberStep,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.utils import make_kill_signal
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


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

    @observe(name="Autofix - Coding Step")
    @sentry_sdk.trace
    @inject
    def _invoke(self, app_config: AppConfig = injected, **kwargs):
        super()._invoke()

        if not self.request.initial_memory:
            # Only clear when not a rethink/continue
            self.context.event_manager.clear_file_changes()

        self.logger.info("Executing Autofix - Coding Step")

        self.context.event_manager.send_coding_start()

        state = self.context.state.get()
        root_cause, root_cause_extra_instruction = state.get_selected_root_cause()
        if not root_cause:
            raise ValueError("Root cause analysis must be performed before coding")

        solution, coding_mode = state.get_selected_solution()
        if not solution:
            raise ValueError("Solution must be found before coding")

        if not self.request.initial_memory:
            self.context.event_manager.add_log(
                f"Coding up a {'fix' if coding_mode == 'fix' else 'test' if coding_mode == 'test' else 'fix and a test'} for this issue..."
            )
        else:
            self.context.event_manager.add_log("Continuing to code...")

        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        summary = state.request.issue_summary
        if not summary:
            summary = self.context.get_issue_summary()

        CodingComponent(self.context).invoke(
            CodingRequest(
                event_details=event_details,
                root_cause=root_cause,
                solution=solution,
                original_instruction=state.request.instruction,
                root_cause_extra_instruction=root_cause_extra_instruction,
                summary=summary,
                initial_memory=self.request.initial_memory,
                profile=state.request.profile,
                mode=coding_mode if coding_mode else "fix",
            )
        )

        state = self.context.state.get()
        if all(not codebase.file_changes for codebase in state.codebases.values()):
            memory = self.context.get_memory("code")
            termination_reason = memory[-1].content if memory else None
            if termination_reason:
                self.context.event_manager.send_coding_result(termination_reason=termination_reason)
                return
            else:
                raise ValueError("No file changes from coding agent")
        if state.steps[-1].status == AutofixStatus.WAITING_FOR_USER_RESPONSE:
            return
        if make_kill_signal() in state.signals:
            return

        self.context.event_manager.send_coding_result()

        # confidence evaluation
        if not self.context.state.get().request.options.disable_interactivity:
            run_memory = self.context.get_memory("code")
            confidence_output = ConfidenceComponent(self.context).invoke(
                ConfidenceRequest(
                    run_memory=run_memory,
                    step_goal_description="implementing the solution and drafting a PR",
                    next_step_goal_description="opening the PR for review by the team",
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

        pr_to_comment_on = state.request.options.comment_on_pr_with_url
        self.next(
            AutofixChangeDescriberStep.get_signature(
                AutofixChangeDescriberRequest(
                    **self.step_request_fields,
                    pr_to_comment_on=pr_to_comment_on,
                ),
            ),
            queue=app_config.CELERY_WORKER_QUEUE,
        )
