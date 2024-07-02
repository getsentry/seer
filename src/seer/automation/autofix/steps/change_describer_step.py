from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import app as celery_app
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


class AutofixChangeDescriberRequest(PipelineStepTaskRequest):
    pass


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

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> AutofixChangeDescriberRequest:
        return AutofixChangeDescriberRequest.model_validate(request)

    @staticmethod
    def get_task():
        return autofix_change_describer_task

    @observe(name="Autofix – Change Describer Step")
    @ai_track(description="Autofix - Change Describer Step")
    def _invoke(self, **kwargs):
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
                            hint="Describe the code changes in the following branch for a pull request.",
                            change_dump=diff_str,
                        )
                    )

                    change = CodebaseChange(
                        repo_id=1,
                        repo_external_id=codebase_state.repo_external_id,
                        repo_name=repo_definition.full_name,
                        title=change_description.title if change_description else "Code Changes",
                        description=change_description.description if change_description else "",
                        diff=diff,
                        diff_str=diff_str,
                    )

                    codebase_changes.append(change)

        self.context.event_manager.send_execution_complete(codebase_changes)
