import dataclasses
import logging
import time
from typing import cast

from seer.automation.autofix.components.coding.models import CodingOutput
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisOutput
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    ChangeDetails,
    ChangesStep,
    CodebaseChange,
    CodeContextRootCauseSelection,
    CustomRootCauseSelection,
    DefaultStep,
    ProgressItem,
    ProgressType,
    RootCauseStep,
    Step,
)
from seer.automation.models import FileChange
from seer.automation.state import State
from seer.automation.utils import make_kill_signal

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AutofixEventManager:
    state: State[AutofixContinuation]

    @property
    def root_cause_analysis_processing_step(self) -> DefaultStep:
        return DefaultStep(
            key="root_cause_analysis_processing",
            title="Analyzing the Issue",
        )

    @property
    def root_cause_analysis_step(self) -> RootCauseStep:
        return RootCauseStep(
            key="root_cause_analysis",
            title="Root Cause Analysis",
        )

    @property
    def plan_step(self) -> DefaultStep:
        return DefaultStep(
            key="plan",
            title="Creating Fix",
        )

    @property
    def changes_step(self) -> ChangesStep:
        return ChangesStep(
            key="changes",
            title="Code Changes",
        )

    def restart_step(self, step: Step):
        with self.state.update() as cur:
            cur_step = cur.find_or_add(step)
            cur_step.status = AutofixStatus.PROCESSING
            cur_step.progress = []
            cur_step.clear_output_stream()
            cur_step.completedMessage = None  # type: ignore[assignment]
            cur.status = AutofixStatus.PROCESSING
            cur.mark_triggered()

    def send_root_cause_analysis_will_start(self):
        with self.state.update() as cur:
            step = cur.add_step(self.root_cause_analysis_processing_step)
            step.status = (
                AutofixStatus.PROCESSING
            )  # We want it to be spinning on the UI, not a grey pending, so we just make it processing.

    def send_root_cause_analysis_start(self):
        with self.state.update() as cur:
            root_cause_step = cur.find_step(key=self.root_cause_analysis_processing_step.key)

            if not root_cause_step or root_cause_step.status != AutofixStatus.PROCESSING:
                root_cause_step = cur.add_step(self.root_cause_analysis_processing_step)

            root_cause_step.status = AutofixStatus.PROCESSING
            cur.make_step_latest(root_cause_step)

            cur.status = AutofixStatus.PROCESSING

    def send_root_cause_analysis_result(self, root_cause_output: RootCauseAnalysisOutput):
        with self.state.update() as cur:
            root_cause_processing_step = cur.find_or_add(self.root_cause_analysis_processing_step)
            root_cause_processing_step.status = AutofixStatus.COMPLETED
            root_cause_step = cur.find_or_add(self.root_cause_analysis_step)
            if root_cause_output.causes:
                root_cause_step.status = AutofixStatus.COMPLETED
                root_cause_step.causes = root_cause_output.causes

                cur.status = AutofixStatus.NEED_MORE_INFORMATION
            else:
                root_cause_step.status = AutofixStatus.ERROR
                cur.status = AutofixStatus.ERROR
                root_cause_step.termination_reason = root_cause_output.termination_reason

    def set_selected_root_cause(self, payload: AutofixRootCauseUpdatePayload):
        root_cause_selection: CustomRootCauseSelection | CodeContextRootCauseSelection | None = None
        if payload.custom_root_cause:
            root_cause_selection = CustomRootCauseSelection(
                custom_root_cause=payload.custom_root_cause,
            )
        elif payload.cause_id is not None:
            root_cause_selection = CodeContextRootCauseSelection(
                cause_id=payload.cause_id,
                instruction=payload.instruction,
            )

        if root_cause_selection is None:
            raise ValueError("Invalid root cause update payload")

        with self.state.update() as cur:
            root_cause_step = cur.find_or_add(self.root_cause_analysis_step)
            root_cause_step.selection = root_cause_selection
            cur.delete_steps_after(root_cause_step, include_current=False)

            cur.status = AutofixStatus.PROCESSING

    def send_coding_start(self):
        with self.state.update() as cur:
            latest_changes_step = cur.find_step(key=self.changes_step.key)
            if (
                not latest_changes_step
                or latest_changes_step.status != AutofixStatus.PROCESSING
                or latest_changes_step.id != cur.steps[-1].id
            ):
                latest_changes_step = cur.add_step(self.changes_step)

            latest_changes_step = cast(ChangesStep, latest_changes_step)

            # Create the initial codebase changes for each repo, if not already present
            for repo in cur.request.repos:
                if repo.external_id not in latest_changes_step.codebase_changes:
                    latest_changes_step.codebase_changes[repo.external_id] = CodebaseChange(
                        repo_name=repo.full_name,
                        repo_external_id=repo.external_id,
                        file_changes=[],
                    )

            latest_changes_step.status = AutofixStatus.PROCESSING

            cur.status = AutofixStatus.PROCESSING

    def send_coding_result(self, result: CodingOutput | None):
        with self.state.update() as cur:
            latest_changes_step = cur.find_or_add(self.changes_step)
            latest_changes_step.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR

            cur.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR

    def set_change_details(self, repo_external_id: str, change_details: ChangeDetails):
        with self.state.update() as cur:
            latest_changes_step = cur.find_or_add(self.changes_step)
            latest_changes_step = cast(ChangesStep, latest_changes_step)
            latest_changes_step.codebase_changes[repo_external_id].details = change_details

    def send_coding_complete(self):
        with self.state.update() as cur:
            cur.mark_running_steps_completed()

            changes_step = cur.find_step(key=self.changes_step.key)
            if not changes_step:
                raise ValueError("Changes step not found")

            changes_step.status = AutofixStatus.COMPLETED

            cur.status = AutofixStatus.COMPLETED

    def add_log(self, message: str):
        with self.state.update() as cur:
            if cur.steps:
                step = cur.steps[-1]
                step.progress.append(
                    ProgressItem(
                        message=message,
                        type=ProgressType.INFO,
                    )
                )

    def ask_user_question(self, question: str):
        with self.state.update() as cur:
            step = cur.steps[-1]
            step.status = AutofixStatus.WAITING_FOR_USER_RESPONSE
            step.progress.append(
                ProgressItem(
                    message=question,
                    type=ProgressType.INFO,
                )
            )

            cur.status = AutofixStatus.WAITING_FOR_USER_RESPONSE

    def add_user_message(self, message: str, memory_index: int):
        with self.state.update() as cur:
            last_step = cur.steps[-1]
            if not isinstance(last_step, DefaultStep):
                last_step = cur.add_step(self.plan_step)

            last_step = cast(DefaultStep, last_step)

            last_step.insights.append(
                InsightSharingOutput(
                    insight=message,
                    justification="USER",
                    codebase_context=[],
                    stacktrace_context=[],
                    breadcrumb_context=[],
                    generated_at_memory_index=memory_index,
                )
            )

    def append_file_change(self, repo_external_id: str, file_change: FileChange):
        with self.state.update() as cur:
            last_changes_step = cur.find_or_add(self.changes_step)
            if last_changes_step.id != cur.steps[-1].id:
                last_changes_step = cur.add_step(self.changes_step)

            if not last_changes_step:
                raise ValueError("Last plan step not found")

            last_changes_step = cast(ChangesStep, last_changes_step)

            # For backwards compatibility, we append to the codebase state's file changes too (for now)
            if (
                cur.codebases
                and cur.codebases[repo_external_id]
                and cur.codebases[repo_external_id].file_changes
            ):
                cur.codebases[repo_external_id].file_changes.append(file_change)

            if repo_external_id not in last_changes_step.codebase_changes:
                raise ValueError(f"Codebase changes for repo {repo_external_id} not found")

            last_changes_step.codebase_changes[repo_external_id].file_changes.append(file_change)

    def on_error(
        self, error_msg: str = "Something went wrong", should_completely_error: bool = True
    ):
        with self.state.update() as cur:
            cur.mark_running_steps_errored()
            cur.set_last_step_completed_message(error_msg)

            if should_completely_error:
                cur.status = AutofixStatus.ERROR

    def reset_steps_to_point(
        self, last_step_to_retain_index: int, last_insight_to_retain_index: int | None
    ) -> bool:
        with self.state.update() as cur:
            cur.kill_all_processing_steps()  # mark any processing steps for killing
            cur.delete_all_steps_after_index(
                last_step_to_retain_index
            )  # delete all steps after specified step
            step = cur.find_step(
                index=last_step_to_retain_index
            )  # delete all insights after specified insight
            if isinstance(step, DefaultStep):
                step.insights = (
                    step.insights[: last_insight_to_retain_index + 1]
                    if last_insight_to_retain_index is not None
                    else []
                )
                cur.steps[-1] = step

        count = 0
        while make_kill_signal() in self.state.get().signals:
            time.sleep(0.5)  # wait for all steps to be killed
            count += 1
            if count > 5:
                return False  # could not kill steps
        return True  # successfully killed steps
