import enum
import logging
from typing import Literal, Optional

from pydantic import BaseModel

from seer.automation.autofix.models import AutofixOutput, PlanningOutput, ProblemDiscoveryResult
from seer.rpc import RpcClient


class Status(enum.Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    CANCELLED = "CANCELLED"


logger = logging.getLogger("autofix")


class Step(BaseModel):
    id: str
    index: int
    description: Optional[str] = None
    title: str
    children: list["Step"] = []
    status: Status


class AutofixEventManager:
    steps: list[Step] = []

    def __init__(self, rpc_client: RpcClient, issue_id: int):
        self.rpc_client = rpc_client
        self.issue_id = issue_id

    def _send_steps_update(self, status: Status):
        self.rpc_client.call(
            "on_autofix_step_update",
            issue_id=self.issue_id,
            status=status,
            steps=[step.model_dump() for step in self.steps],
        )

    def send_no_stacktrace_error(self):
        self.steps = [
            Step(
                id="problem_discovery",
                index=0,
                title="Preliminary Assessment",
                status=Status.ERROR,
                description="Error: Cannot fix issues without a stacktrace.",
            )
        ]

        self._send_steps_update(Status.ERROR)

    def send_initial_steps(self):
        self.steps = [
            Step(
                id="problem_discovery",
                index=0,
                title="Preliminary Assessment",
                status=Status.PROCESSING,
            )
        ]

        self._send_steps_update(Status.PROCESSING)
        logger.debug("Sent initial steps")

    def send_problem_discovery_result(self, result: ProblemDiscoveryResult):
        problem_discovery_step = next(step for step in self.steps if step.id == "problem_discovery")
        problem_discovery_step.description = result.description
        problem_discovery_step.status = Status.COMPLETED
        self.steps = [
            problem_discovery_step,
        ]

        self._send_steps_update(
            Status.PROCESSING if result.status == "CONTINUE" else Status.COMPLETED
        )
        logger.debug(f"Sent problem discovery result: {result}")

    def send_codebase_creation_message(self):
        self.steps.extend(
            [
                Step(
                    id="codebase_indexing",
                    index=1,
                    title="Codebase Indexing",
                    description="This will take longer than usual because this is the first time you've run Autofix on this codebase.",
                    status=Status.PROCESSING,
                ),
                Step(
                    id="plan",
                    index=2,
                    title="Execution Plan",
                    status=Status.PENDING,
                ),
            ]
        )

        self._send_steps_update(Status.PROCESSING)

    def send_codebase_indexing_message(self):
        self.steps.extend(
            [
                Step(
                    id="codebase_indexing",
                    index=1,
                    title="Codebase Indexing",
                    status=Status.PROCESSING,
                ),
                Step(
                    id="plan",
                    index=2,
                    title="Execution Plan",
                    status=Status.PENDING,
                ),
            ]
        )

        self._send_steps_update(Status.PROCESSING)

    def send_codebase_creation_skip(self):
        self.steps.extend(
            [
                Step(
                    id="plan",
                    index=1,
                    title="Execution Plan",
                    status=Status.PENDING,
                ),
            ]
        )

        self._send_steps_update(Status.PROCESSING)

    def send_codebase_indexing_result(
        self, status: Literal[Status.COMPLETED, Status.ERROR, Status.CANCELLED]
    ):
        # Update the status of step 2 to COMPLETED and step 3 to PROCESSING
        for step in self.steps:
            if step.id == "codebase_indexing":
                step.status = status
                step.description = None
            elif step.id == "plan":
                step.status = Status.PROCESSING if status != Status.ERROR else Status.CANCELLED

        self._send_steps_update(Status.PROCESSING if status != Status.ERROR else Status.ERROR)
        logger.debug(f"Sent codebase indexing result: {status}")

    def send_planning_result(self, result: PlanningOutput | None):
        plan_step = next(step for step in self.steps if step.id == "plan")
        plan_step.status = Status.PROCESSING if result else Status.ERROR
        if result:
            plan_step.title = "Execute Plan"

            plan_step.children = [
                Step(
                    id=str(plan_step.id),
                    index=1,
                    title=plan_step.title,
                    status=Status.PENDING,
                )
                for plan_step in result.steps
            ]

            if len(plan_step.children) > 0:
                plan_step.children[0].status = Status.PROCESSING

        self._send_steps_update(Status.PROCESSING if result else Status.ERROR)
        logger.debug(f"Sent planning result: {result}")

    def send_execution_step_start(self, execution_id: int):
        plan_step = next(step for step in self.steps if step.id == "plan")
        execution_step = next(
            child for child in plan_step.children if child.id == str(execution_id)
        )
        execution_step.status = Status.PROCESSING

        self._send_steps_update(Status.PROCESSING)
        logger.debug(f"Sent execution step start: {execution_id}")

    def send_execution_step_result(
        self, execution_id: int, status: Literal[Status.COMPLETED, Status.ERROR]
    ):
        plan_step = next(step for step in self.steps if step.id == "plan")
        execution_step = next(
            child for child in plan_step.children if child.id == str(execution_id)
        )
        execution_step.status = status

        self._send_steps_update(Status.PROCESSING if status == Status.COMPLETED else Status.ERROR)
        logger.debug(f"Sent execution step result: {execution_id} {status}")

    def mark_all_steps_completed(self):
        for step in self.steps:
            step.status = Status.COMPLETED

    def mark_running_steps_errored(self):
        for step in self.steps:
            if step.status == Status.PROCESSING:
                step.status = Status.ERROR
                for substep in step.children:
                    if substep.status == Status.PROCESSING:
                        substep.status = Status.ERROR
                    if substep.status == Status.PENDING:
                        substep.status = Status.CANCELLED

    def send_autofix_complete(self, fix: AutofixOutput | None):
        if fix:
            self.mark_all_steps_completed()

        self.rpc_client.call(
            "on_autofix_complete",
            issue_id=self.issue_id,
            status=Status.COMPLETED if fix else Status.ERROR,
            steps=[step.model_dump() for step in self.steps],
            fix=fix.model_dump() if fix else None,
        )
        logger.debug(f"Sent autofix complete: {fix}")
