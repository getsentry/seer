import datetime
import enum
import hashlib
import os
from typing import Annotated, Any, Literal, Optional, Union

from johen import gen
from johen.examples import Examples
from johen.generators import specialized
from pydantic import BaseModel, ConfigDict, Field, field_validator

from seer.automation.agent.models import Usage
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.config import AUTOFIX_HARD_TIME_OUT_SECS
from seer.automation.models import FileChange, FilePatch, IssueDetails, RepoDefinition


class FileChangeError(Exception):
    pass


class ProgressType(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    NEED_MORE_INFORMATION = "NEED_MORE_INFORMATION"
    USER_RESPONSE = "USER_RESPONSE"


class ProgressItem(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    message: str
    type: ProgressType
    data: Any = None


class AutofixStatus(enum.Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    NEED_MORE_INFORMATION = "NEED_MORE_INFORMATION"
    CANCELLED = "CANCELLED"

    @classmethod
    def terminal(cls) -> "frozenset[AutofixStatus]":
        return frozenset((cls.COMPLETED, cls.ERROR, cls.CANCELLED))


class ProblemDiscoveryResult(BaseModel):
    status: Literal["CONTINUE", "CANCELLED"]
    description: str
    reasoning: str


class AutofixUserDetails(BaseModel):
    id: Annotated[int, Examples(specialized.unsigned_ints)]
    display_name: str


class AutofixExecutionComplete(BaseModel):
    title: str
    description: str
    diff: Optional[list[FilePatch]] = []
    diff_str: Optional[str] = None


class AutofixOutput(AutofixExecutionComplete):
    usage: Usage


class AutofixEndpointResponse(BaseModel):
    started: bool


class CustomRootCauseSelection(BaseModel):
    custom_root_cause: str


class SuggestedFixRootCauseSelection(BaseModel):
    cause_id: int
    fix_id: int


RootCauseSelection = Union[CustomRootCauseSelection, SuggestedFixRootCauseSelection]


class CommittedPullRequestDetails(BaseModel):
    pr_number: int
    pr_url: str
    pr_id: Optional[int] = None


class CodebaseChange(BaseModel):
    repo_id: int
    repo_name: str
    title: str
    description: str
    diff: list[FilePatch] = []
    diff_str: Optional[str] = None
    pull_request: Optional[CommittedPullRequestDetails] = None


class StepType(str, enum.Enum):
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    CHANGES = "changes"
    USER_RESPONSE = "user_response"
    DEFAULT = "default"


class BaseStep(BaseModel):
    id: str = Field(
        default_factory=lambda: hashlib.sha1(os.urandom(16)).hexdigest()
    )  # Unique identifier for this step
    key: str  # Identifier for a type of step
    title: str
    type: StepType = StepType.DEFAULT

    status: AutofixStatus = AutofixStatus.PENDING

    index: int = -1
    progress: list["ProgressItem | Step"] = Field(default_factory=list)
    completedMessage: Optional[str] = None

    def find_child(self, *, id: str) -> "Step | None":
        for step in self.progress:
            if (
                isinstance(step, (DefaultStep, RootCauseStep, ChangesStep, UserResponseStep))
                and step.id == id
            ):
                return step
        return None

    def find_or_add_child(self, base_step: "Step") -> "Step":
        existing = self.find_child(id=base_step.id)
        if existing:
            return existing

        base_step = base_step.model_copy()
        base_step.index = len(self.progress)
        self.progress.append(base_step)
        return base_step


class DefaultStep(BaseStep):
    type: Literal[StepType.DEFAULT] = StepType.DEFAULT


class RootCauseStep(BaseStep):
    type: Literal[StepType.ROOT_CAUSE_ANALYSIS] = StepType.ROOT_CAUSE_ANALYSIS

    causes: list[RootCauseAnalysisItem] = []
    selection: RootCauseSelection | None = None


class ChangesStep(BaseStep):
    type: Literal[StepType.CHANGES] = StepType.CHANGES

    changes: list[CodebaseChange]


class UserResponseStep(BaseStep):
    type: Literal[StepType.USER_RESPONSE] = StepType.USER_RESPONSE

    text: str
    user_id: int


Step = Union[DefaultStep, RootCauseStep, ChangesStep, UserResponseStep]


class CodebaseState(BaseModel):
    repo_id: int
    namespace_id: int
    file_changes: list[FileChange] = []


class AutofixStateOptions(BaseModel):
    iterative_feedback: bool | None = False


class AutofixGroupState(BaseModel):
    run_id: int = -1
    steps: list[Step] = Field(default_factory=list)
    status: AutofixStatus = AutofixStatus.PENDING
    codebases: dict[int, CodebaseState] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    last_triggered_at: Optional[
        Annotated[datetime.datetime, Examples(datetime.datetime.now() for _ in gen)]
    ] = None
    updated_at: Optional[
        Annotated[datetime.datetime, Examples(datetime.datetime.now() for _ in gen)]
    ] = None
    completed_at: datetime.datetime | None = None
    signals: list[str] = Field(default_factory=list)
    actor_ids: list[int] = Field(default_factory=list)
    options: AutofixStateOptions | None = Field(default=AutofixStateOptions())


class AutofixStateRequest(BaseModel):
    group_id: int


class AutofixPrIdRequest(BaseModel):
    provider: str
    pr_id: int


class AutofixStateResponse(BaseModel):
    group_id: Optional[int]
    state: Optional[dict]


class AutofixCompleteArgs(BaseModel):
    issue_id: int
    status: AutofixStatus
    steps: list[Step]
    fix: AutofixOutput | None


class AutofixStepUpdateArgs(BaseModel):
    issue_id: int
    status: AutofixStatus
    steps: list[Step]


class AutofixRequest(BaseModel):
    organization_id: Annotated[int, Examples(specialized.unsigned_ints)]
    project_id: Annotated[int, Examples(specialized.unsigned_ints)]
    repos: list[RepoDefinition]
    issue: IssueDetails
    invoking_user: Optional[AutofixUserDetails] = None
    base_commit_sha: Optional[
        Annotated[str, Examples(hashlib.sha1(s).hexdigest() for s in specialized.byte_strings)]
    ] = None
    instruction: Optional[str] = Field(default=None, validation_alias="additional_context")

    @property
    def process_request_name(self) -> str:
        return f"autofix:{self.organization_id}:{self.issue.id}"

    @field_validator("repos", mode="after")
    @classmethod
    def validate_repo_duplicates(cls, repos):
        if isinstance(repos, list):
            # Check for duplicates by comparing lengths after converting to a set
            if len(set(repos)) != len(repos):
                raise ValueError("Duplicate repos detected in the request.")
            return repos

        raise ValueError("Not a list of repos.")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AutofixUpdateType(str, enum.Enum):
    SELECT_ROOT_CAUSE = "select_root_cause"
    CREATE_PR = "create_pr"
    INSTRUCTION = "instruction"


class AutofixRootCauseUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.SELECT_ROOT_CAUSE]
    cause_id: int | None = None
    fix_id: int | None = None
    custom_root_cause: str | None = None
    is_retry: bool | None = False


class AutofixCreatePrUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.CREATE_PR]
    repo_id: int | None = None


class AutofixTextInstruction(BaseModel):
    type: Literal["text"]
    text: str


class AutofixInstructionPayload(BaseModel):
    type: Literal[AutofixUpdateType.INSTRUCTION]
    content: AutofixTextInstruction


class AutofixUpdateRequest(BaseModel):
    run_id: int
    invoking_user: AutofixUserDetails
    payload: Union[
        AutofixRootCauseUpdatePayload, AutofixCreatePrUpdatePayload, AutofixInstructionPayload
    ] = Field(discriminator="type")


class AutofixContinuation(AutofixGroupState):
    request: AutofixRequest

    def find_step(self, *, key: str | None = None, id: str | None = None) -> Step | None:
        for step in reversed(self.steps):
            if step.key == key or step.id == id:
                return step
        return None

    def add_step(self, base_step: Step):
        base_step = base_step.model_copy()
        base_step.index = len(self.steps)
        self.steps.append(base_step)

        return base_step

    def find_or_add(self, base_step: Step) -> Step:
        existing = self.find_step(key=base_step.key)

        if existing:
            return existing

        return self.add_step(base_step)

    def last_or_add(self, base_step: Step) -> Step:
        last_step = self.steps[-1]
        if last_step and (last_step.id == base_step.id or last_step.key == base_step.key):
            return last_step

        return self.add_step(base_step)

    def make_step_latest(self, step: Step):
        if step in self.steps:
            self.steps.remove(step)
            self.steps.append(step)

    def mark_all_steps_completed(self):
        for step in self.steps:
            step.status = AutofixStatus.COMPLETED

    def _mark_steps_errored(self, status_condition: AutofixStatus):
        did_mark = False
        for step in self.steps:
            if step.status == status_condition:
                step.status = AutofixStatus.ERROR
                did_mark = True
                for substep in step.progress:
                    if isinstance(substep, (DefaultStep, RootCauseStep, ChangesStep)):
                        if substep.status == AutofixStatus.PROCESSING:
                            substep.status = AutofixStatus.ERROR
                        if substep.status == AutofixStatus.PENDING:
                            substep.status = AutofixStatus.CANCELLED

        return did_mark

    def mark_running_steps_errored(self):
        did_mark = self._mark_steps_errored(AutofixStatus.PROCESSING)

        if not did_mark:
            self._mark_steps_errored(AutofixStatus.PENDING)

    def set_last_step_completed_message(self, message: str):
        if self.steps:
            self.steps[-1].completedMessage = message

    def get_selected_root_cause_and_fix(self) -> RootCauseAnalysisItem | str | None:
        root_cause_step = self.find_step(key="root_cause_analysis")
        if root_cause_step and isinstance(root_cause_step, RootCauseStep):
            if root_cause_step.selection:
                if isinstance(root_cause_step.selection, SuggestedFixRootCauseSelection):
                    cause = next(
                        cause
                        for cause in root_cause_step.causes
                        if cause.id == root_cause_step.selection.cause_id
                    )

                    if cause.suggested_fixes:
                        fix = next(
                            fix
                            for fix in cause.suggested_fixes
                            if fix.id == root_cause_step.selection.fix_id
                        )

                        cause = cause.model_copy()

                        cause.suggested_fixes = [fix]

                        return cause
                elif isinstance(root_cause_step.selection, CustomRootCauseSelection):
                    return root_cause_step.selection.custom_root_cause
        return None

    def mark_triggered(self):
        self.last_triggered_at = datetime.datetime.now()

    def mark_updated(self):
        self.updated_at = datetime.datetime.now()

    @property
    def is_running(self):
        return self.status == AutofixStatus.PROCESSING or self.status == AutofixStatus.PENDING

    @property
    def has_timed_out(self, now: datetime.datetime | None = None) -> bool:
        if self.is_running and self.last_triggered_at:
            if now is None:
                now = datetime.datetime.now()
            return (
                self.last_triggered_at + datetime.timedelta(seconds=AUTOFIX_HARD_TIME_OUT_SECS)
                < now
            )
        return False
