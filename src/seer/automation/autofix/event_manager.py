import enum
import logging
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel

from seer.automation.autofix.models import AutofixOutput, PlanningOutput, ProblemDiscoveryResult
from seer.rpc import RpcClient


class AutofixStatus(enum.Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    CANCELLED = "CANCELLED"


class ProgressType(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    NEED_MORE_INFORMATION = "NEED_MORE_INFORMATION"
    USER_RESPONSE = "USER_RESPONSE"


logger = logging.getLogger("autofix")


class ProgressItem(BaseModel):
    timestamp: str
    message: str
    type: ProgressType
    data: Any = None


class Step(BaseModel):
    id: str
    index: int
    completedMessage: Optional[str] = None
    title: str

    status: AutofixStatus

    progress: list["ProgressItem | Step"] = []


class AutofixEventManager:
    steps: list[Step] = []

    def __init__(self, rpc_client: RpcClient, issue_id: int):
        self.rpc_client = rpc_client
        self.issue_id = issue_id

    def _send_steps_update(self, status: AutofixStatus):
        self.rpc_client.call(
            "on_autofix_step_update",
            issue_id=self.issue_id,
            status=status,
            steps=[step.model_dump() for step in self.steps],
        )

    def _get_step(self, step_id: str, steps: Optional[list[ProgressItem | Step]] = None) -> Step:
        steps_to_use = steps or self.steps
        step = next(step for step in steps_to_use if isinstance(step, Step) and step.id == step_id)

        return step

    def send_no_stacktrace_error(self):
        self.steps = [
            Step(
                id="problem_discovery",
                index=0,
                title="Preliminary Assessment",
                status=AutofixStatus.ERROR,
                completedMessage="Error: Cannot fix issues without a stacktrace.",
            )
        ]

        self._send_steps_update(AutofixStatus.ERROR)

    def send_initial_steps(self):
        self.steps = [
            Step(
                id="problem_discovery",
                index=0,
                title="Preliminary Assessment",
                status=AutofixStatus.PROCESSING,
            )
        ]

        self._send_steps_update(AutofixStatus.PROCESSING)
        logger.debug("Sent initial steps")

    def send_problem_discovery_result(self, result: ProblemDiscoveryResult):
        problem_discovery_step = self._get_step("problem_discovery")
        problem_discovery_step.progress = [
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
        problem_discovery_step.completedMessage = result.description
        problem_discovery_step.status = AutofixStatus.COMPLETED
        self.steps = [
            problem_discovery_step,
            Step(
                id="codebase_indexing",
                index=1,
                title="Codebase Indexing",
                status=AutofixStatus.PENDING,
                progress=[],
            ),
            Step(
                id="plan",
                index=2,
                title="Execution Plan",
                status=AutofixStatus.PENDING,
                progress=[],
            ),
        ]

        self._send_steps_update(
            AutofixStatus.PROCESSING if result.status == "CONTINUE" else AutofixStatus.COMPLETED
        )
        logger.debug(f"Sent problem discovery result: {result}")

    def send_codebase_indexing_repo_check_message(self, repo_full_name: str):
        indexing_step = self._get_step("codebase_indexing")

        indexing_step.status = AutofixStatus.PROCESSING
        indexing_step.progress.append(
            ProgressItem(
                timestamp=datetime.now().isoformat(),
                message=f"Checking if {repo_full_name} is indexed...",
                type=ProgressType.INFO,
            )
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_indexing_repo_exists_message(self, repo_full_name: str):
        indexing_step = self._get_step("codebase_indexing")

        indexing_step.status = AutofixStatus.PROCESSING
        indexing_step.progress.append(
            ProgressItem(
                timestamp=datetime.now().isoformat(),
                message=f"{repo_full_name} is indexed.",
                type=ProgressType.INFO,
            )
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_index_creation_message(self, repo_full_name: str):
        indexing_step = self._get_step("codebase_indexing")

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

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_index_creation_complete_message(self, repo_full_name: str):
        indexing_step = self._get_step("codebase_indexing")

        indexing_step.status = AutofixStatus.PROCESSING
        indexing_step.progress.append(
            ProgressItem(
                timestamp=datetime.now().isoformat(),
                message=f"Indexing {repo_full_name} complete.",
                type=ProgressType.INFO,
            )
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_index_up_to_date_message(self, repo_full_name: str):
        indexing_step = self._get_step("codebase_indexing")

        indexing_step.status = AutofixStatus.PROCESSING
        indexing_step.progress.append(
            ProgressItem(
                timestamp=datetime.now().isoformat(),
                message=f"{repo_full_name} is up to date.",
                type=ProgressType.INFO,
            )
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_index_update_wait_message(self, repo_full_name: str):
        indexing_step = self._get_step("codebase_indexing")

        indexing_step.status = AutofixStatus.PROCESSING
        indexing_step.progress.append(
            ProgressItem(
                timestamp=datetime.now().isoformat(),
                message=f"{repo_full_name} needs to be updated. Waiting for the update to complete...",
                type=ProgressType.INFO,
            )
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_index_update_scheduled_message(self, repo_full_name: str):
        indexing_step = self._get_step("codebase_indexing")

        indexing_step.status = AutofixStatus.PROCESSING
        indexing_step.progress.append(
            ProgressItem(
                timestamp=datetime.now().isoformat(),
                message=f"{repo_full_name} will be updated in the background.",
                type=ProgressType.INFO,
            )
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_indexing_result(
        self, status: Literal[AutofixStatus.COMPLETED, AutofixStatus.ERROR, AutofixStatus.CANCELLED]
    ):
        # Update the status of step 2 to COMPLETED and step 3 to PROCESSING
        for step in self.steps:
            if step.id == "codebase_indexing":
                step.status = status
                step.completedMessage = None
                step.progress.append(
                    ProgressItem(
                        timestamp=datetime.now().isoformat(),
                        message="Codebase indexing complete.",
                        type=ProgressType.INFO,
                    )
                )
            elif step.id == "plan":
                step.status = (
                    AutofixStatus.PROCESSING
                    if status != AutofixStatus.ERROR
                    else AutofixStatus.CANCELLED
                )

        self._send_steps_update(
            AutofixStatus.PROCESSING if status != AutofixStatus.ERROR else AutofixStatus.ERROR
        )
        logger.debug(f"Sent codebase indexing result: {status}")

    def send_planning_result(self, result: PlanningOutput | None):
        plan_step = next(step for step in self.steps if step.id == "plan")
        plan_step.status = AutofixStatus.PROCESSING if result else AutofixStatus.ERROR
        if result:
            plan_step.title = "Execute Plan"

            plan_step.progress = [
                Step(
                    id=str(plan_step.id),
                    index=1,
                    title=plan_step.title,
                    status=AutofixStatus.PENDING,
                    progress=[],
                )
                for plan_step in result.steps
            ]

            if len(plan_step.progress) > 0:
                plan_step.progress[0].status = AutofixStatus.PROCESSING

        self._send_steps_update(AutofixStatus.PROCESSING if result else AutofixStatus.ERROR)
        logger.debug(f"Sent planning result: {result}")

    def send_execution_step_start(self, execution_id: int):
        plan_step = self._get_step("plan")
        execution_step = self._get_step(str(execution_id), plan_step.progress)
        execution_step.status = AutofixStatus.PROCESSING

        self._send_steps_update(AutofixStatus.PROCESSING)
        logger.debug(f"Sent execution step start: {execution_id}")

    def send_execution_step_result(
        self, execution_id: int, status: Literal[AutofixStatus.COMPLETED, AutofixStatus.ERROR]
    ):
        plan_step = self._get_step("plan")
        execution_step = self._get_step(str(execution_id), plan_step.progress)
        execution_step.status = status

        self._send_steps_update(
            AutofixStatus.PROCESSING if status == AutofixStatus.COMPLETED else AutofixStatus.ERROR
        )
        logger.debug(f"Sent execution step result: {execution_id} {status}")

    def mark_all_steps_completed(self):
        for step in self.steps:
            step.status = AutofixStatus.COMPLETED

    def mark_running_steps_errored(self):
        for step in self.steps:
            if step.status == AutofixStatus.PROCESSING:
                step.status = AutofixStatus.ERROR
                for substep in step.progress:
                    if isinstance(substep, Step):
                        if substep.status == AutofixStatus.PROCESSING:
                            substep.status = AutofixStatus.ERROR
                        if substep.status == AutofixStatus.PENDING:
                            substep.status = AutofixStatus.CANCELLED

    def send_autofix_complete(self, fix: AutofixOutput | None):
        if fix:
            self.mark_all_steps_completed()

        self.rpc_client.call(
            "on_autofix_complete",
            issue_id=self.issue_id,
            status=AutofixStatus.COMPLETED if fix else AutofixStatus.ERROR,
            steps=[step.model_dump() for step in self.steps],
            fix=fix.model_dump() if fix else None,
        )
        logger.debug(f"Sent autofix complete: {fix}")
