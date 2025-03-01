import logging
from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import celery_app
from seer.automation.autofix.components.change_describer import (
    ChangeDescriptionComponent,
    ChangeDescriptionRequest,
)
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import CodebaseChange
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.pipeline import PipelineStepTaskRequest

logger = logging.getLogger(__name__)


class AutofixChangeDescriberRequest(PipelineStepTaskRequest):
    pr_to_comment_on: str | None = None


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def autofix_change_describer_task(*args, request: dict[str, Any]):
    AutofixChangeDescriberStep(request).invoke()


class AutofixChangeDescriberStep(AutofixPipelineStep):
    """
    This class represents the execution pipeline in the autofix system. It is responsible for
    executing the fixes suggested by the planning component based on the root cause analysis.
    """

    name = "AutofixChangeDescriberStep"
    request: AutofixChangeDescriberRequest

    max_retries = 1

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> AutofixChangeDescriberRequest:
        return AutofixChangeDescriberRequest.model_validate(request)

    @staticmethod
    def get_task():
        return autofix_change_describer_task

    @observe(name="Autofix – Change Describer Step")
    @ai_track(description="Autofix - Change Describer Step")
    def _invoke(self, **kwargs):
        self.context.event_manager.add_log("Writing a commit message, of course...")
        # Get the diff and PR details for each codebase.
        change_describer = ChangeDescriptionComponent(self.context)
        codebase_changes: list[CodebaseChange] = []
        cur_state = self.context.state.get()
        for codebase_state in cur_state.codebases.values():
            if codebase_state.file_changes:
                if not codebase_state.repo_external_id:
                    raise ValueError("Codebase state does not have a repo external id")

                repo_definition = self.context.repos_by_key().get(codebase_state.repo_external_id)

                if not repo_definition:
                    raise ValueError(
                        f"Could not find repo definition for external id {codebase_state.repo_external_id}"
                    )

                diff, diff_str = self.context.make_file_patches(
                    codebase_state.file_changes, repo_definition.full_name
                )

                if diff:
                    change_description = change_describer.invoke(
                        ChangeDescriptionRequest(
                            change_dump=diff_str,
                        )
                    )

                    change = CodebaseChange(
                        repo_external_id=codebase_state.repo_external_id,
                        repo_name=repo_definition.full_name,
                        title=change_description.title if change_description else "Code Changes",
                        description=change_description.description if change_description else "",
                        diff=diff,
                        diff_str=diff_str,
                        draft_branch_name=(
                            change_description.branch_name if change_description else None
                        ),
                    )

                    codebase_changes.append(change)

        self.context.event_manager.send_complete(codebase_changes)
        if codebase_changes:
            self.context.event_manager.add_log(
                "Here are Autofix's suggested changes to fix the issue."
            )
        else:
            self.context.event_manager.add_log(
                "Autofix couldn't find any changes to make. Maybe help Autofix rethink by editing a card above?"
            )
            logger.exception("Autofix couldn't find a fix.")

        # GitHub Copilot can automatically make a PR after the coding step
        if self.request.pr_to_comment_on:
            for repo in cur_state.request.repos:
                self.context.commit_changes(
                    repo_external_id=repo.external_id,
                    pr_to_comment_on_url=self.request.pr_to_comment_on,
                    make_pr=True,
                )
