from typing import Any

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import app as celery_app
from seer.automation.autofix.components.change_describer import (
    ChangeDescriptionComponent,
    ChangeDescriptionRequest,
)
from seer.automation.autofix.components.planner.models import PlanningOutput
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

    @observe(name="Change Describer")
    @ai_track(description="Autofix - Change Describer")
    def _invoke(self, **kwargs):
        # Get the diff and PR details for each codebase.
        change_describer = ChangeDescriptionComponent(self.context)
        codebase_changes: list[CodebaseChange] = []
        for codebase in self.context.codebases.values():
            diff, diff_str = codebase.get_file_patches()

            if diff:
                change_description = change_describer.invoke(
                    ChangeDescriptionRequest(
                        hint="Describe the code changes in the following branch for a pull request.",
                        change_dump=diff_str,
                    )
                )

                change = CodebaseChange(
                    repo_id=codebase.repo_info.id,
                    repo_name=codebase.repo_info.external_slug,
                    title=change_description.title if change_description else "Code Changes",
                    description=change_description.description if change_description else "",
                    diff=diff,
                    diff_str=diff_str,
                )

                codebase_changes.append(change)

        self.context.event_manager.send_execution_complete(codebase_changes)
