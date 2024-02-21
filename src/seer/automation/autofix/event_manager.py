import enum
import json
from enum import Enum

class AutoFixJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
import logging
from typing import Literal, Optional

from pydantic import BaseModel

from seer.automation.autofix.models import AutofixOutput, PlanningOutput, ProblemDiscoveryResult
from seer.rpc import RpcClient


class AutofixStatus(enum.Enum):
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
    status: AutofixStatus


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

    def send_no_stacktrace_error(self):
        self.steps = [
            Step(
                id="problem_discovery",
                index=0,
                title="Preliminary Assessment",
                status=AutofixStatus.ERROR,
                description="Error: Cannot fix issues without a stacktrace.",
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
        problem_discovery_step = next(step for step in self.steps if step.id == "problem_discovery")
        problem_discovery_step.description = result.description
        problem_discovery_step.status = AutofixStatus.COMPLETED
        self.steps = [
            problem_discovery_step,
        ]

        self._send_steps_update(
            AutofixStatus.PROCESSING if result.status == "CONTINUE" else AutofixStatus.COMPLETED
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
                    status=AutofixStatus.PROCESSING,
                ),
                Step(
                    id="plan",
                    index=2,
                    title="Execution Plan",
                    status=AutofixStatus.PENDING,
                ),
            ]
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_indexing_message(self):
        self.steps.extend(
            [
                Step(
                    id="codebase_indexing",
                    index=1,
                    title="Codebase Indexing",
                    status=AutofixStatus.PROCESSING,
                ),
                Step(
                    id="plan",
                    index=2,
                    title="Execution Plan",
                    status=AutofixStatus.PENDING,
                ),
            ]
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_creation_skip(self):
        self.steps.extend(
            [
                Step(
                    id="plan",
                    index=1,
                    title="Execution Plan",
                    status=AutofixStatus.PENDING,
                ),
            ]
        )

        self._send_steps_update(AutofixStatus.PROCESSING)

    def send_codebase_indexing_result(
        self, status: Literal[AutofixStatus.COMPLETED, AutofixStatus.ERROR, AutofixStatus.CANCELLED]
    ):
        # Update the status of step 2 to COMPLETED and step 3 to PROCESSING
        for step in self.steps:
            if step.id == "codebase_indexing":
                step.status = status
                step.description = None
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

            plan_step.children = [
                Step(
                    id=str(plan_step.id),
                    index=1,
                    title=plan_step.title,
                    status=AutofixStatus.PENDING,
                )
                for plan_step in result.steps
            ]

            if len(plan_step.children) > 0:
                plan_step.children[0].status = AutofixStatus.PROCESSING

        self._send_steps_update(AutofixStatus.PROCESSING if result else AutofixStatus.ERROR)
        logger.debug(f"Sent planning result: {result}")

    def send_execution_step_start(self, execution_id: int):
        plan_step = next(step for step in self.steps if step.id == "plan")
        execution_step = next(
            child for child in plan_step.children if child.id == str(execution_id)
        )
        execution_step.status = AutofixStatus.PROCESSING

        self._send_steps_update(AutofixStatus.PROCESSING)
        logger.debug(f"Sent execution step start: {execution_id}")

    def send_execution_step_result(
        self, execution_id: int, status: Literal[AutofixStatus.COMPLETED, AutofixStatus.ERROR]
    ):
        plan_step = next(step for step in self.steps if step.id == "plan")
        execution_step = next(
            child for child in plan_step.children if child.id == str(execution_id)
        )
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
                for substep in step.children:
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
