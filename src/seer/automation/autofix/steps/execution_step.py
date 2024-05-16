from typing import Any

from sentry_sdk.ai.monitoring import ai_track

from celery_app.app import app as celery_app
from seer.automation.autofix.components.change_describer import (
    ChangeDescriptionComponent,
    ChangeDescriptionRequest,
)
from seer.automation.autofix.components.executor.component import ExecutorComponent
from seer.automation.autofix.components.executor.models import ExecutorRequest
from seer.automation.autofix.components.planner.component import PlanningComponent
from seer.automation.autofix.components.planner.models import (
    CreateFilePromptXml,
    PlanningRequest,
    ReplaceCodePromptXml,
)
from seer.automation.autofix.components.retriever import RetrieverComponent, RetrieverRequest
from seer.automation.autofix.config import (
    AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
from seer.automation.autofix.models import AutofixStatus, CodebaseChange
from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.models import EventDetails
from seer.automation.pipeline import PipelineStepTaskRequest


class AutofixExecutionStepRequest(PipelineStepTaskRequest):
    pass


@celery_app.task(
    time_limit=AUTOFIX_EXECUTION_HARD_TIME_LIMIT_SECS,
    soft_time_limit=AUTOFIX_EXECUTION_SOFT_TIME_LIMIT_SECS,
)
def autofix_execution_task(*args, request: dict[str, Any]):
    AutofixExecutionStep(request).invoke()


class AutofixExecutionStep(AutofixPipelineStep):
    """
    This class represents the execution pipeline in the autofix system. It is responsible for
    executing the fixes suggested by the planning component based on the root cause analysis.
    """

    name = "AutofixExecutionStep"

    @staticmethod
    def _instantiate_request(request: dict[str, Any]) -> AutofixExecutionStepRequest:
        return AutofixExecutionStepRequest.model_validate(request)

    @staticmethod
    def get_task():
        return autofix_execution_task

    @ai_track(description="Autofix - Execution")
    def _invoke(self, **kwargs):
        self.context.event_manager.send_codebase_indexing_complete_if_exists()
        self.context.event_manager.send_planning_start()

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
                instruction=state.request.instruction,
            )
        )

        self.context.event_manager.send_planning_result(planning_output)

        if not planning_output:
            return

        retriever = RetrieverComponent(self.context)
        executor = ExecutorComponent(self.context)
        for i, task in enumerate(planning_output.tasks):
            self.context.event_manager.send_execution_step_start(i)
            self._run_executor_with_retriever(retriever, executor, task, event_details)
            self.context.event_manager.send_execution_step_result(i, AutofixStatus.COMPLETED)

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

    @ai_track(description="Executor with Retriever")
    def _run_executor_with_retriever(
        self,
        retriever: RetrieverComponent,
        executor: ExecutorComponent,
        task: ReplaceCodePromptXml | CreateFilePromptXml,
        event_details: EventDetails,
    ):
        retriever_output = retriever.invoke(RetrieverRequest(text=task.to_prompt_str()))

        executor.invoke(
            ExecutorRequest(
                event_details=event_details,
                retriever_dump=(
                    retriever_output.to_xml().to_prompt_str() if retriever_output else None
                ),
                task=task.to_prompt_str(),
            )
        )
