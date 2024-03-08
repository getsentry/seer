import dataclasses
import logging
from datetime import datetime
from typing import Literal, Optional

from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixOutput,
    AutofixStatus,
    PlanningOutput,
    ProblemDiscoveryResult,
    ProgressItem,
    ProgressType,
    Step,
)
from seer.automation.state import State

logger = logging.getLogger("autofix")


@dataclasses.dataclass
class AutofixEventManager:
    state: State[AutofixContinuation]

    @property
    def problem_discovery_step(self) -> Step:
        return Step(
            id="problem_discovery",
            title="Preliminary Assessment",
        )

    def send_no_stacktrace_error(self):
        with self.state.update() as cur:
            step = self.problem_discovery_step
            step.description = "Error: Cannot fix issues without a stacktrace."
            cur.steps = [step]
            cur.status = step.status = AutofixStatus.ERROR

    def send_initial_steps(self):
        with self.state.update() as cur:
            step = self.problem_discovery_step
            cur.steps = [step]
            step.status = AutofixStatus.PROCESSING
            cur.status = AutofixStatus.PROCESSING

    @property
    def indexing_step(self) -> Step:
        return Step(
            id="codebase_indexing",
            title="Codebase Indexing",
        )

    @property
    def plan_step(self) -> Step:
        return Step(
            id="plan",
            title="Execution Plan",
        )

    def send_problem_discovery_result(self, result: ProblemDiscoveryResult):
        with self.state.update() as cur:
            problem_discovery_step = cur.find_or_add(self.problem_discovery_step)
            problem_discovery_step.description = result.description
            problem_discovery_step.status = AutofixStatus.COMPLETED
            problem_discovery_step.progress.extend(
                [
                    ProgressItem(
                        timestamp=datetime.now().isoformat(),
                        message=result.reasoning,
                        type=ProgressType.INFO,
                    ),
                    ProgressItem(
                        timestamp=datetime.now().isoformat(),
                        message=result.description,
                        type=ProgressType.INFO,
                    ),
                ]
            )
            problem_discovery_step.completedMessage = result.description

            cur.find_or_add(self.indexing_step)
            cur.find_or_add(self.plan_step)

            cur.status = (
                AutofixStatus.PROCESSING if result.status == "CONTINUE" else AutofixStatus.COMPLETED
            )

    def send_codebase_indexing_repo_check_message(self, repo_full_name: str):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)
            indexing_step.status = AutofixStatus.PROCESSING
            indexing_step.progress.append(
                ProgressItem(
                    timestamp=datetime.now().isoformat(),
                    message=f"Checking if {repo_full_name} is indexed...",
                    type=ProgressType.INFO,
                )
            )

    def send_codebase_indexing_repo_exists_message(self, repo_full_name: str):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)

            indexing_step.status = AutofixStatus.PROCESSING
            indexing_step.progress.append(
                ProgressItem(
                    timestamp=datetime.now().isoformat(),
                    message=f"{repo_full_name} is indexed.",
                    type=ProgressType.INFO,
                )
            )
            cur.status = AutofixStatus.PROCESSING

    def send_codebase_index_creation_message(self, repo_full_name: str):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)

            indexing_step.status = AutofixStatus.PROCESSING
            indexing_step.progress.extend(
                [
                    ProgressItem(
                        timestamp=datetime.now().isoformat(),
                        message=f"Indexing {repo_full_name}...",
                        type=ProgressType.INFO,
                    ),
                    ProgressItem(
                        timestamp=datetime.now().isoformat(),
                        message=f"Because this is the first time indexing {repo_full_name}, this may take a while...",
                        type=ProgressType.INFO,
                    ),
                ]
            )

            cur.status = AutofixStatus.PROCESSING

    def send_codebase_index_creation_complete_message(self, repo_full_name: str):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)

            indexing_step.status = AutofixStatus.PROCESSING
            indexing_step.progress.append(
                ProgressItem(
                    timestamp=datetime.now().isoformat(),
                    message=f"Indexing {repo_full_name} complete.",
                    type=ProgressType.INFO,
                )
            )

            cur.status = AutofixStatus.PROCESSING

    def send_codebase_index_up_to_date_message(self, repo_full_name: str):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)

            indexing_step.status = AutofixStatus.PROCESSING
            indexing_step.progress.append(
                ProgressItem(
                    timestamp=datetime.now().isoformat(),
                    message=f"{repo_full_name} is up to date.",
                    type=ProgressType.INFO,
                )
            )

            cur.status = AutofixStatus.PROCESSING

    def send_codebase_index_update_wait_message(self, repo_full_name: str):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)

            indexing_step.status = AutofixStatus.PROCESSING
            indexing_step.progress.append(
                ProgressItem(
                    timestamp=datetime.now().isoformat(),
                    message=f"{repo_full_name} needs to be updated. Waiting for the update to complete...",
                    type=ProgressType.INFO,
                )
            )

            cur.status = AutofixStatus.PROCESSING

    def send_codebase_index_update_scheduled_message(self, repo_full_name: str):
        with self.state.update() as cur:
            indexing_step = cur.find_or_add(self.indexing_step)
            indexing_step.status = AutofixStatus.PROCESSING
            indexing_step.progress.append(
                ProgressItem(
                    timestamp=datetime.now().isoformat(),
                    message=f"{repo_full_name} will be updated in the background.",
                    type=ProgressType.INFO,
                )
            )

            cur.status = AutofixStatus.PROCESSING

    def send_codebase_indexing_result(
        self, status: Literal[AutofixStatus.COMPLETED, AutofixStatus.ERROR, AutofixStatus.CANCELLED]
    ):
        with self.state.update() as cur:
            indexing_step = cur.find_step(id=self.indexing_step.id)
            if indexing_step:
                indexing_step.status = status
                indexing_step.completedMessage = None
                indexing_step.progress.append(
                    ProgressItem(
                        timestamp=datetime.now().isoformat(),
                        message="Codebase indexing complete.",
                        type=ProgressType.INFO,
                    )
                )

            plan_step = cur.find_step(id=self.plan_step.id)
            if plan_step:
                plan_step.status = (
                    AutofixStatus.PROCESSING
                    if status != AutofixStatus.ERROR
                    else AutofixStatus.CANCELLED
                )

            cur.status = (
                AutofixStatus.PROCESSING if status != AutofixStatus.ERROR else AutofixStatus.ERROR
            )

    def send_planning_result(self, result: PlanningOutput | None):
        with self.state.update() as cur:
            plan_step = cur.find_or_add(self.plan_step)
            plan_step.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR
            if result:
                plan_step.title = "Execute Plan"
                for child_step in result.steps:
                    plan_step.find_or_add_child(
                        Step(
                            id=str(child_step.id),
                            title=child_step.title,
                        )
                    )

            if len(plan_step.progress) > 0:
                plan_step.progress[0].status = AutofixStatus.PROCESSING

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

    def mark_all_steps_completed(self, state: AutofixContinuation):
        for step in state.steps:
            step.status = AutofixStatus.COMPLETED

    def mark_running_steps_errored(self, state: AutofixContinuation):
        for step in state.steps:
            if step.status == AutofixStatus.PROCESSING:
                step.status = AutofixStatus.ERROR
                for substep in step.progress:
                    if isinstance(substep, Step):
                        if substep.status == AutofixStatus.PROCESSING:
                            substep.status = AutofixStatus.ERROR
                        if substep.status == AutofixStatus.PENDING:
                            substep.status = AutofixStatus.CANCELLED

    def send_autofix_complete(self, fix: AutofixOutput | None):
        with self.state.update() as cur:
            if fix:
                self.mark_all_steps_completed(cur)
            else:
                self.mark_running_steps_errored(cur)
            cur.status = AutofixStatus.COMPLETED if fix else AutofixStatus.ERROR
