import dataclasses

import sentry_sdk
from langsmith import traceable

from seer.automation.autofix.autofix_context import AutofixContext
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
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisRequest
from seer.automation.autofix.models import AutofixStatus, CodebaseChange
from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.models import UpdateCodebaseTaskRequest
from seer.automation.codebase.tasks import update_codebase_index
from seer.automation.models import EventDetails
from seer.automation.pipeline import Pipeline, PipelineSideEffect


@dataclasses.dataclass
class CreateAnyMissingCodebaseIndexesSideEffect(PipelineSideEffect):
    """
    Side effect that creates any missing codebase indexes for the repositories in the context.

    Typically will not be called because codebase indexing would be performed separate from the autofix run.
    """

    context: AutofixContext

    def invoke(self):
        if self.context.has_missing_codebase_indexes():
            self.context.create_missing_codebase_indexes()


@dataclasses.dataclass
class CheckCodebaseForUpdatesSideEffect(PipelineSideEffect):
    """
    A side effect in the pipeline that checks for updates in the codebase.

    This class is responsible for checking if the codebase is behind its remote counterpart,
    and if so, it triggers the necessary actions to update the codebase index. It uses the
    event details from the context to determine if the update is necessary based on the
    presence of stacktrace files in the diff. If an immediate update is not required,
    it schedules an update for later.
    """

    context: AutofixContext

    def invoke(self):
        if self.context.has_codebase_indexing_run():
            autofix_logger.info("Codebase indexing already performed, update side effect skipped.")
            return

        event_details = EventDetails.from_event(self.context.state.get().request.issue.events[0])

        for repo_id, codebase in self.context.codebases.items():
            if codebase.is_behind():
                if self.context.diff_contains_stacktrace_files(repo_id, event_details):
                    autofix_logger.debug(
                        f"Waiting for codebase index update for repo {codebase.repo_info.external_slug}"
                    )
                    self.context.event_manager.send_codebase_indexing_start()
                    self.context.event_manager.add_log(
                        f"Creating codebase index for repo: {codebase.repo_info.external_slug}"
                    )
                    with sentry_sdk.start_span(
                        op="seer.automation.autofix.codebase_index.update",
                        description="Update codebase index",
                    ) as span:
                        span.set_tag("repo", codebase.repo_info.external_slug)
                        codebase.update()
                    autofix_logger.debug(f"Codebase index updated")
                    self.context.event_manager.add_log(
                        f"Created codebase index for repo: {codebase.repo_info.external_slug}"
                    )
                else:
                    update_codebase_index.apply_async(
                        (UpdateCodebaseTaskRequest(repo_id=repo_id).model_dump(),),
                        countdown=10 * 60,
                    )  # 10 minutes
                    autofix_logger.info(f"Codebase indexing scheduled for later")

        self.context.event_manager.send_codebase_indexing_complete_if_exists()


class AutofixRootCause(Pipeline):
    """
    This class represents the root cause analysis pipeline in the autofix system. It is responsible for
    analyzing the root cause of issues detected in the codebase and suggesting potential fixes.
    """

    context: AutofixContext
    side_effects: list[PipelineSideEffect]

    def __init__(self, context: AutofixContext):
        super().__init__(context)
        self.side_effects = [
            CreateAnyMissingCodebaseIndexesSideEffect(context),
            CheckCodebaseForUpdatesSideEffect(context),
        ]

    @traceable(name="Root Cause", tags=["autofix:v2"])
    def _invoke(self):
        self.context.event_manager.send_root_cause_analysis_start()

        if self.context.has_missing_codebase_indexes():
            self.context.create_missing_codebase_indexes()

        state = self.context.state.get()
        event_details = EventDetails.from_event(state.request.issue.events[0])
        self.context.process_event_paths(event_details)

        root_cause_output = RootCauseAnalysisComponent(self.context).invoke(
            RootCauseAnalysisRequest(
                event_details=event_details,
                instruction=state.request.instruction,
            )
        )

        self.context.event_manager.send_root_cause_analysis_result(root_cause_output)

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()


class AutofixExecution(Pipeline):
    """
    This class represents the execution pipeline in the autofix system. It is responsible for
    executing the fixes suggested by the planning component based on the root cause analysis.
    """

    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)
        self.side_effects = [CheckCodebaseForUpdatesSideEffect(context)]

    @traceable(name="Execution", tags=["autofix:v2"])
    def _invoke(self):
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

    def _handle_exception(self, exception: Exception):
        self.context.event_manager.on_error()

    @traceable(name="Executor with Retriever", run_type="llm")
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
