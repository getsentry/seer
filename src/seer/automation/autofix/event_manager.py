import dataclasses
import logging
import time

from seer.automation.autofix.components.coding.models import CodingOutput
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisOutput
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
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
            changes=[],
        )

    def migrate_step_keys(self):
        # TODO: Remove this we no longer need the backwards compatibility.
        with self.state.update() as cur:
            for step in cur.steps:
                step.ensure_uuid_id()

    def restart_step(self, step: Step):
        with self.state.update() as cur:
            cur_step = cur.find_or_add(step)
            cur_step.status = AutofixStatus.PROCESSING
            cur_step.progress = []
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
            cur.clear_file_changes()

            cur.status = AutofixStatus.PROCESSING

    def send_coding_start(self):
        with self.state.update() as cur:
            plan_step = cur.find_step(key=self.plan_step.key)
            if not plan_step or plan_step.status != AutofixStatus.PROCESSING:
                plan_step = cur.add_step(self.plan_step)

            plan_step.status = AutofixStatus.PROCESSING

            cur.status = AutofixStatus.PROCESSING

    def send_coding_result(self, result: CodingOutput | None):
        with self.state.update() as cur:
            plan_step = cur.find_or_add(self.plan_step)
            plan_step.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR

            cur.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR

    def send_coding_complete(self, codebase_changes: list[CodebaseChange]):
        with self.state.update() as cur:
            cur.mark_running_steps_completed()

            changes_step = cur.find_or_add(self.changes_step)
            changes_step.status = AutofixStatus.COMPLETED
            changes_step.changes = codebase_changes

            cur.status = AutofixStatus.COMPLETED

    def add_log(self, message: str):
        with self.state.update() as cur:
            if cur.steps:
                step = cur.steps[-1]

                # If the current step is the planning step, and an execution step is running, we log it there instead.
                if step.id == self.plan_step.id and step.progress:
                    # select the first execution step that is processing
                    execution_step = next(
                        (
                            step
                            for step in step.progress
                            if isinstance(step, DefaultStep)
                            and step.status == AutofixStatus.PROCESSING
                        ),
                        None,
                    )

                    if execution_step:
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

    def on_error(
        self, error_msg: str = "Something went wrong", should_completely_error: bool = True
    ):
        with self.state.update() as cur:
            cur.mark_running_steps_errored()
            cur.set_last_step_completed_message(error_msg)

            if should_completely_error:
                cur.status = AutofixStatus.ERROR

    def clear_file_changes(self):
        with self.state.update() as cur:
            cur.clear_file_changes()

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
