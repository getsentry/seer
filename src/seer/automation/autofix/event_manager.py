import dataclasses
import logging
from typing import Literal

from seer.automation.autofix.components.planner.models import PlanningOutput
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisOutput
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixStatus,
    ChangesStep,
    CodebaseChange,
    DefaultStep,
    ProgressItem,
    ProgressType,
    RootCauseSelection,
    RootCauseStep,
)
from seer.automation.state import State

logger = logging.getLogger("autofix")


@dataclasses.dataclass
class AutofixEventManager:
    state: State[AutofixContinuation]

    @property
    def root_cause_analysis_processing_step(self) -> DefaultStep:
        return DefaultStep(
            id="root_cause_analysis_processing",
            title="Analyze Issue",
        )

    @property
    def root_cause_analysis_step(self) -> RootCauseStep:
        return RootCauseStep(
            id="root_cause_analysis",
            title="Root Cause Analysis",
        )

    @property
    def indexing_step(self) -> DefaultStep:
        return DefaultStep(
            id="codebase_indexing",
            title="Codebase Indexing",
        )

    @property
    def plan_step(self) -> DefaultStep:
        return DefaultStep(
            id="plan",
            title="Create Fix",
        )

    @property
    def changes_step(self) -> ChangesStep:
        return ChangesStep(
            id="changes",
            title="Changes",
            changes=[],
        )

    def send_root_cause_analysis_pending(self):
        with self.state.update() as cur:
            root_cause_step = cur.find_or_add(self.root_cause_analysis_processing_step)
            root_cause_step.status = AutofixStatus.PENDING

            cur.status = AutofixStatus.PROCESSING

    def send_root_cause_analysis_start(self):
        with self.state.update() as cur:
            root_cause_step = cur.find_or_add(self.root_cause_analysis_processing_step)
            root_cause_step.status = AutofixStatus.PROCESSING
            cur.make_step_latest(root_cause_step)

            cur.status = AutofixStatus.PROCESSING

    def send_root_cause_analysis_result(self, root_cause_output: RootCauseAnalysisOutput | None):
        with self.state.update() as cur:
            root_cause_processing_step = cur.find_or_add(self.root_cause_analysis_processing_step)
            root_cause_processing_step.status = AutofixStatus.COMPLETED
            root_cause_step = cur.find_or_add(self.root_cause_analysis_step)
            if root_cause_output and root_cause_output.causes:
                root_cause_step.status = AutofixStatus.COMPLETED
                root_cause_step.causes = root_cause_output.causes

                cur.status = AutofixStatus.NEED_MORE_INFORMATION
            else:
                root_cause_step.status = AutofixStatus.ERROR
                cur.status = AutofixStatus.ERROR

    def send_codebase_indexing_start(self):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)
            indexing_step.status = AutofixStatus.PROCESSING

            cur.status = AutofixStatus.PROCESSING

    def send_codebase_indexing_complete_if_exists(self):
        with self.state.update() as cur:
            indexing_step = cur.find_step(id=self.indexing_step.id)

            if indexing_step:
                indexing_step.status = AutofixStatus.COMPLETED

    def set_selected_root_cause(self, selection: RootCauseSelection):
        with self.state.update() as cur:
            root_cause_step = cur.find_or_add(self.root_cause_analysis_step)
            root_cause_step.selection = selection

            cur.status = AutofixStatus.PROCESSING

    def send_planning_pending(self):
        with self.state.update() as cur:
            root_cause_step = cur.find_or_add(self.plan_step)
            root_cause_step.status = AutofixStatus.PENDING

            cur.status = AutofixStatus.PROCESSING

    def send_planning_start(self):
        with self.state.update() as cur:
            plan_step = cur.find_or_add(self.plan_step)
            plan_step.status = AutofixStatus.PROCESSING

            cur.status = AutofixStatus.PROCESSING

    def send_planning_result(self, result: PlanningOutput | None):
        with self.state.update() as cur:
            plan_step = cur.find_or_add(self.plan_step)
            plan_step.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR

            if result:
                for i, child_step in enumerate(result.tasks):
                    step = plan_step.find_or_add_child(
                        DefaultStep(
                            id=str(i),
                            title=child_step.commit_message,
                        )
                    )

                    step.status = AutofixStatus.PROCESSING

            cur.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR

    def send_execution_step_start(self, execution_id: int):
        with self.state.update() as cur:
            plan_step = cur.find_or_add(self.plan_step)
            execution_step = plan_step.find_child(id=str(execution_id))
            if execution_step:
                execution_step.status = AutofixStatus.PROCESSING
            cur.status = AutofixStatus.PROCESSING

    def send_execution_step_result(
        self, execution_id: int, status: Literal[AutofixStatus.COMPLETED, AutofixStatus.ERROR]
    ):
        with self.state.update() as cur:
            plan_step = cur.find_or_add(self.plan_step)
            execution_step = plan_step.find_child(id=str(execution_id))
            if execution_step:
                execution_step.status = status

            cur.status = (
                AutofixStatus.PROCESSING
                if status == AutofixStatus.COMPLETED
                else AutofixStatus.ERROR
            )

    def send_execution_complete(self, codebase_changes: list[CodebaseChange]):
        with self.state.update() as cur:
            cur.mark_all_steps_completed()

            changes_step = cur.find_or_add(self.changes_step)
            changes_step.status = AutofixStatus.COMPLETED
            changes_step.changes = codebase_changes

            cur.status = AutofixStatus.COMPLETED

    def send_pr_creation_start(self):
        with self.state.update() as cur:
            changes_step = cur.find_or_add(self.changes_step)
            changes_step.status = AutofixStatus.PROCESSING
            cur.status = AutofixStatus.PROCESSING

    def send_pr_creation_complete(self):
        with self.state.update() as cur:
            changes_step = cur.find_or_add(self.changes_step)
            changes_step.status = AutofixStatus.COMPLETED
            cur.status = AutofixStatus.COMPLETED

    def add_log(self, message: str):
        with self.state.update() as cur:
            step = cur.steps[-1]
            if step.status == AutofixStatus.PROCESSING:
                # If the current step is the planning step, and an execution step is running, we log it there instead.
                if step.id == self.plan_step.id and step.progress:
                    execution_step = step.progress[-1]
                    if (
                        isinstance(execution_step, DefaultStep)
                        and execution_step.status == AutofixStatus.PROCESSING
                    ):
                        execution_step.progress.append(
                            ProgressItem(
                                message=message,
                                type=ProgressType.INFO,
                            )
                        )
                        return
                step.progress.append(
                    ProgressItem(
                        message=message,
                        type=ProgressType.INFO,
                    )
                )

    def on_error(self, error_msg: str = "Something went wrong"):
        with self.state.update() as cur:
            cur.mark_running_steps_errored()
            cur.set_last_step_completed_message(error_msg)

            cur.status = AutofixStatus.ERROR
