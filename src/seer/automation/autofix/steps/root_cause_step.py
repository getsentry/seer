import uuid
from typing import Any

import sentry_sdk
from langfuse.decorators import observe

from celery_app.app import celery_app
from seer.automation.agent.models import Message
from seer.automation.autofix.components.confidence import ConfidenceComponent, ConfidenceRequest
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisOutput,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.config import (
    AUTOFIX_ROOT_CAUSE_HARD_TIME_LIMIT_SECS,
    AUTOFIX_ROOT_CAUSE_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import (
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    CommentThread,
)
from seer.automation.autofix.steps.solution_step import (
    AutofixSolutionStep,
    AutofixSolutionStepRequest,
)
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.autofix.utils import find_original_snippet
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest
from seer.automation.utils import make_kill_signal
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


class RootCauseStepRequest(PipelineStepTaskRequest):
    initial_memory: list[Message] = []


@celery_app.task(
    time_limit=AUTOFIX_ROOT_CAUSE_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_ROOT_CAUSE_SOFT_TIME_LIMIT_SECS,
    acks_late=True,
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
    @sentry_sdk.trace
    @inject
    def _invoke(self, app_config: AppConfig = injected, **kwargs):
        super()._invoke()

        self.context.event_manager.send_root_cause_analysis_start()

        if self.request.is_retry:
            self.context.event_manager.add_log("Something broke. Re-analyzing from scratch...")
        elif not self.request.initial_memory:
            self.context.event_manager.add_log("Figuring out the root cause...")
        else:
            self.context.event_manager.add_log("Going back to the drawing board...")

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
                profile=state.request.profile,
                trace_tree=state.request.trace_tree,
            )
        )

        state = self.context.state.get()
        if state.steps and state.steps[-1].status == AutofixStatus.WAITING_FOR_USER_RESPONSE:
            return
        if make_kill_signal() in state.signals:
            return

        # create URLs to relevant code snippets
        reproduction_urls = self._get_reproduction_urls(root_cause_output)
        root_cause_output.causes[0].reproduction_urls = reproduction_urls

        self.context.event_manager.send_root_cause_analysis_result(root_cause_output)

        # confidence evaluation
        if not self.context.state.get().request.options.disable_interactivity:
            run_memory = self.context.get_memory("root_cause_analysis")
            confidence_output = ConfidenceComponent(self.context).invoke(
                ConfidenceRequest(
                    run_memory=run_memory,
                    step_goal_description="root cause analysis",
                    next_step_goal_description="figuring out a solution",
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

        self.context.event_manager.add_log(
            "Here is Autofix's proposed root cause."
            if root_cause_output.termination_reason is None and root_cause_output.causes
            else "Autofix couldn't find the root cause. Maybe help Autofix rethink by editing a card above?"
        )

        # GitHub Copilot can comment on a provided PR with the root cause analysis
        pr_to_comment_on = state.request.options.comment_on_pr_with_url
        if pr_to_comment_on:
            causes = root_cause_output.causes
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

        # Early return if no causes were found
        if not root_cause_output.causes:
            return

        # Only proceed with solution step if we have root causes
        self.context.event_manager.set_selected_root_cause(
            AutofixRootCauseUpdatePayload(
                cause_id=root_cause_output.causes[0].id,
            )
        )
        self.next(
            AutofixSolutionStep.get_signature(
                AutofixSolutionStepRequest(
                    **self.step_request_fields,
                    initial_memory=[],
                ),
            ),
            queue=app_config.CELERY_WORKER_QUEUE,
        )

    def _get_reproduction_urls(self, root_cause_output: RootCauseAnalysisOutput):
        reproduction_urls: list[str | None] = []
        repro_timeline = root_cause_output.causes[0].root_cause_reproduction
        if not repro_timeline:
            return []
        for i, timeline_item in enumerate(repro_timeline):
            reproduction_urls.append(None)
            relevant_code = timeline_item.relevant_code_file
            if not relevant_code:
                continue
            repo_name = self.context.autocorrect_repo_name(relevant_code.repo_name)
            if not repo_name:
                continue
            file_name = self.context.autocorrect_file_path(
                path=relevant_code.file_path, repo_name=repo_name
            )
            if not file_name:
                continue

            repo_client = self.context.get_repo_client(repo_name)

            full_snippet = timeline_item.code_snippet_and_analysis
            # Extract code snippet between triple backticks
            code_snippet = None
            parts = full_snippet.split("```")
            if len(parts) > 2:
                code_snippet = parts[1].split("```")[0]
                if "\n" in code_snippet:
                    code_snippet = "\n".join(
                        code_snippet.split("\n")[1:]
                    )  # remove the first line because it's often something like ```python
            start_line = None
            end_line = None
            if code_snippet:
                file_content = self.context.get_file_contents(file_name, repo_name)
                if file_content:
                    result = find_original_snippet(
                        code_snippet, file_content, initial_line_threshold=0.5, threshold=0.5
                    )
                    if result:
                        start_line = result[1]
                        end_line = result[2]
            code_url = repo_client.get_file_url(file_name, start_line, end_line)
            reproduction_urls[i] = code_url

        return reproduction_urls
