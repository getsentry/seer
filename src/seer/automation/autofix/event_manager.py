import dataclasses
import logging
import time

from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisOutput
from seer.automation.autofix.components.solution.models import (
    RelevantCodeFileWithUrl,
    SolutionOutput,
    SolutionTimelineEvent,
)
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRootCauseUpdatePayload,
    AutofixSolutionUpdatePayload,
    AutofixStatus,
    ChangesStep,
    CodebaseChange,
    CodeContextRootCauseSelection,
    CustomRootCauseSelection,
    DefaultStep,
    ProgressItem,
    ProgressType,
    RootCauseStep,
    SolutionStep,
    Step,
)
from seer.automation.state import State
from seer.automation.utils import make_kill_signal
from seer.events import SeerEventNames, log_seer_event

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
    def solution_processing_step(self) -> DefaultStep:
        return DefaultStep(
            key="solution_processing",
            title="Planning Solution",
        )

    @property
    def solution_step(self) -> SolutionStep:
        return SolutionStep(
            key="solution",
            title="Solution",
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
        log_payload = None

        with self.state.update() as cur:
            root_cause_step = cur.find_step(key=self.root_cause_analysis_processing_step.key)

            if not root_cause_step or root_cause_step.status != AutofixStatus.PROCESSING:
                root_cause_step = cur.add_step(self.root_cause_analysis_processing_step)

            root_cause_step.status = AutofixStatus.PROCESSING
            cur.make_step_latest(root_cause_step)

            cur.status = AutofixStatus.PROCESSING

            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
                "is_auto_run": cur.request.options.auto_run_source is not None,
                "auto_run_source": cur.request.options.auto_run_source,
            }

        if log_payload:
            log_seer_event(
                SeerEventNames.AUTOFIX_ROOT_CAUSE_STARTED,
                log_payload,
            )

    def send_root_cause_analysis_result(self, root_cause_output: RootCauseAnalysisOutput):
        log_payload = None
        with self.state.update() as cur:
            root_cause_processing_step = cur.find_or_add(self.root_cause_analysis_processing_step)
            root_cause_processing_step.status = AutofixStatus.COMPLETED
            root_cause_step = cur.find_or_add(self.root_cause_analysis_step)

            root_cause_step.status = AutofixStatus.COMPLETED
            cur.status = AutofixStatus.COMPLETED

            if root_cause_output.causes:
                root_cause_step.causes = root_cause_output.causes
            else:
                root_cause_step.termination_reason = root_cause_output.termination_reason

            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
            }

        if log_payload:
            log_seer_event(
                SeerEventNames.AUTOFIX_ROOT_CAUSE_COMPLETED,
                log_payload,
            )

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

    def send_solution_start(self):
        log_payload = None
        with self.state.update() as cur:
            solution_step = cur.find_step(key=self.solution_processing_step.key)
            if not solution_step or solution_step.status != AutofixStatus.PROCESSING:
                solution_step = cur.add_step(self.solution_processing_step)

            solution_step.status = AutofixStatus.PROCESSING
            cur.make_step_latest(solution_step)
            cur.status = AutofixStatus.PROCESSING

            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
            }

        if log_payload:
            log_seer_event(
                SeerEventNames.AUTOFIX_SOLUTION_STARTED,
                log_payload,
            )

    def send_solution_result(
        self, solution_output: SolutionOutput, reproduction_urls: list[str | None]
    ):
        log_payload = None
        with self.state.update() as cur:
            solution_processing_step = cur.find_or_add(self.solution_processing_step)
            solution_processing_step.status = AutofixStatus.COMPLETED
            solution_step = cur.find_or_add(self.solution_step)
            solution_step.status = AutofixStatus.COMPLETED
            solution_step.solution = [
                SolutionTimelineEvent(
                    title=solution_step.title,
                    code_snippet_and_analysis=solution_step.code_snippet_and_analysis,
                    relevant_code_file=(
                        RelevantCodeFileWithUrl(
                            file_path=solution_step.relevant_code_file.file_path,
                            repo_name=solution_step.relevant_code_file.repo_name,
                            url=reproduction_urls[i] if i < len(reproduction_urls) else None,
                        )
                        if solution_step.relevant_code_file
                        else None
                    ),
                    is_most_important_event=solution_step.is_most_important,
                )
                for i, solution_step in enumerate(solution_output.solution_steps)
            ]
            solution_step.description = solution_output.summary
            cur.status = AutofixStatus.NEED_MORE_INFORMATION

            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
            }

        if log_payload:
            log_seer_event(
                SeerEventNames.AUTOFIX_SOLUTION_COMPLETED,
                log_payload,
            )

    def set_selected_solution(self, payload: AutofixSolutionUpdatePayload):
        with self.state.update() as cur:
            solution_step = cur.solution_step

            if not solution_step:
                raise ValueError("Solution step not found to set the selected solution")

            solution_step.custom_solution = (
                payload.custom_solution if payload.custom_solution else None
            )
            original_solution = solution_step.solution
            if payload.solution:
                solution_step.solution = payload.solution
            solution_step.selected_mode = payload.mode
            solution_step.solution_selected = True
            cur.delete_steps_after(solution_step, include_current=False)
            cur.clear_file_changes()

            cur.status = AutofixStatus.PROCESSING

            run_id = cur.run_id
            group_id = cur.request.issue.id
            new_solution = solution_step.solution
            old_solution = original_solution

        # This is here so we can get the original and updated solution for the event.
        # set_selected_solution is called at coding start anyways.
        self._log_coding_start(
            run_id=run_id,
            group_id=group_id,
            new_solution=new_solution,
            original_solution=old_solution,
        )

    def send_coding_start(self):
        with self.state.update() as cur:
            plan_step = cur.find_step(key=self.plan_step.key)
            if not plan_step or plan_step.status != AutofixStatus.PROCESSING:
                plan_step = cur.add_step(self.plan_step)

            plan_step.status = AutofixStatus.PROCESSING

            cur.status = AutofixStatus.PROCESSING

    def send_coding_result(self, termination_reason: str | None = None):
        log_payload = None
        with self.state.update() as cur:
            plan_step = cur.find_or_add(self.plan_step)
            plan_step.status = AutofixStatus.PROCESSING
            cur.status = AutofixStatus.PROCESSING

            if termination_reason:
                changes_step = cur.find_or_add(self.changes_step)
                changes_step.termination_reason = termination_reason
                plan_step.status = AutofixStatus.COMPLETED
                changes_step.status = AutofixStatus.COMPLETED
                cur.status = AutofixStatus.COMPLETED

            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
            }

        if log_payload:
            log_seer_event(
                SeerEventNames.AUTOFIX_CODING_COMPLETED,
                log_payload,
            )

    def send_complete(self, codebase_changes: list[CodebaseChange]):
        log_payload = None
        with self.state.update() as cur:
            cur.mark_running_steps_completed()

            changes_step = cur.find_or_add(self.changes_step)
            changes_step.status = AutofixStatus.COMPLETED
            changes_step.changes = codebase_changes

            cur.status = AutofixStatus.COMPLETED

            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
            }

        if log_payload:
            log_seer_event(
                SeerEventNames.AUTOFIX_COMPLETED,
                log_payload,
            )

    def send_push_changes_start(self):
        with self.state.update() as cur:
            cur.mark_triggered()
            cur.status = AutofixStatus.PROCESSING
            changes_step = cur.changes_step
            if changes_step:
                changes_step.status = AutofixStatus.PROCESSING

    def send_push_changes_result(self):
        with self.state.update() as cur:
            cur.status = AutofixStatus.COMPLETED
            changes_step = cur.changes_step
            if changes_step:
                changes_step.status = AutofixStatus.COMPLETED

    def on_confidence_question(self, question: str):
        cur = self.state.get()
        log_seer_event(
            SeerEventNames.AUTOFIX_ASKED_USER_QUESTION,
            {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
                "question": question,
            },
        )

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
        log_payload = None
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
            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
                "question": question,
            }

        log_seer_event(
            SeerEventNames.AUTOFIX_ASKED_USER_QUESTION,
            log_payload,
        )

    def on_error(
        self, error_msg: str = "Something went wrong", should_completely_error: bool = True
    ):
        log_payload = None
        with self.state.update() as cur:
            cur.mark_running_steps_errored()
            cur.set_last_step_completed_message(error_msg)

            if should_completely_error:
                cur.status = AutofixStatus.ERROR

            current_running_step = None
            if cur.steps:
                current_running_step = cur.steps[-1]

            log_payload = {
                "run_id": cur.run_id,
                "group_id": cur.request.issue.id,
                "error_msg": error_msg,
                "current_running_step": (
                    current_running_step.key if current_running_step else None
                ),
                "should_completely_error": should_completely_error,
            }

            if (
                not should_completely_error and cur.steps
            ):  # delete the last step so it's replaced with the new one upon retry
                cur.steps = cur.steps[:-1]

        if log_payload:
            log_seer_event(
                SeerEventNames.AUTOFIX_RUN_ERROR,
                log_payload,
            )

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
                    and last_insight_to_retain_index >= 0
                    else []
                )
                cur.steps[-1] = step

            # Clear file changes if the coding step is not present.
            if not next((step for step in cur.steps if step.key == self.plan_step.key), None):
                cur.clear_file_changes()

        count = 0
        while make_kill_signal() in self.state.get().signals:
            time.sleep(0.5)  # wait for all steps to be killed
            count += 1
            if count > 5:
                return False  # could not kill steps
        return True  # successfully killed steps

    def _log_coding_start(
        self,
        run_id: int,
        group_id: int,
        new_solution: list[SolutionTimelineEvent],
        original_solution: list[SolutionTimelineEvent],
    ):
        log_seer_event(
            SeerEventNames.AUTOFIX_CODING_STARTED,
            {
                "run_id": run_id,
                "group_id": group_id,
                "has_unit_tests": any(
                    step.timeline_item_type == "repro_test" for step in new_solution
                ),
                "has_removed_steps": any(
                    not any(
                        original_step.title == current_step.title for current_step in new_solution
                    )
                    for original_step in original_solution
                ),
                "has_added_steps": any(
                    not any(
                        solution_step.title == original_step.title
                        for original_step in original_solution
                    )
                    for solution_step in new_solution
                ),
            },
        )

    def send_insight(self, insight_card: InsightSharingOutput, step_id: str | None = None):
        with self.state.update() as cur:
            if insight_card:
                cur_step = cur.find_step(id=step_id) if step_id else cur.steps[-1]

                if not cur_step or not isinstance(cur_step, DefaultStep):
                    logger.exception(
                        f"Cannot add insight to step: step not found or not a DefaultStep. Step key: {cur_step.key if cur_step else 'None'}"
                    )
                    return

                cur_step.insights.append(insight_card)
