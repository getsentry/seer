from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.agent.models import Message
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisRequest
from seer.automation.autofix.config import (
    AUTOFIX_ROOT_CAUSE_HARD_TIME_LIMIT_SECS,
    AUTOFIX_ROOT_CAUSE_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import AutofixStatus
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.utils import make_kill_signal


class RootCauseStepRequest(PipelineStepTaskRequest):
    initial_memory: list[Message] = []


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

        if not self.request.initial_memory:
            self.context.event_manager.add_log("Beginning root cause analysis...")
        else:
            self.context.event_manager.add_log("Continuing to analyze...")

        state = self.context.state.get()
        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        summary = state.request.issue_summary
        if not summary:
            summary = self.context.get_issue_summary()

        root_cause_output = RootCauseAnalysisComponent(self.context).invoke(
            RootCauseAnalysisRequest(
                event_details=event_details,
                instruction=state.request.instruction,
                summary=summary,
                initial_memory=self.request.initial_memory,
            )
        )

        state = self.context.state.get()
        if state.steps and state.steps[-1].status == AutofixStatus.WAITING_FOR_USER_RESPONSE:
            return
        if make_kill_signal() in state.signals:
            return

        self.context.event_manager.send_root_cause_analysis_result(root_cause_output)
        self.context.event_manager.add_log(
            "Here is Autofix's proposed root cause."
            if root_cause_output
            else "Sorry, Autofix couldn't find the root cause."
        )

        # GitHub Copilot can comment on a provided PR with the root cause analysis
        pr_to_comment_on = state.request.options.comment_on_pr_with_url
        if pr_to_comment_on:
            causes = root_cause_output.causes if root_cause_output else []
            cause_string = "Autofix couldn't find a root cause for this issue."
            if causes:
                cause_string = causes[0].to_markdown_string()
            for repo in state.request.repos:
                if (
                    repo.name in pr_to_comment_on and repo.owner in pr_to_comment_on
                ):  # crude check that the repo matches the PR we want to comment on
                    self.context.comment_root_cause_on_pr(
                        pr_url=pr_to_comment_on, repo_definition=repo, root_cause=cause_string
                    )
