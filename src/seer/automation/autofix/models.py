import datetime
import enum
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
    run_id: int


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
    repo_id: int | None = None
    repo_external_id: str | None = None
    repo_name: str
    title: str
    description: str
    diff: list[FilePatch] = []
    diff_str: Optional[str] = None
    pull_request: Optional[CommittedPullRequestDetails] = None


class StepType(str, enum.Enum):
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    CHANGES = "changes"
    DEFAULT = "default"


class BaseStep(BaseModel):
    id: str
    title: str
    type: StepType = StepType.DEFAULT

    status: AutofixStatus = AutofixStatus.PENDING

    index: int = -1
    progress: list["ProgressItem | Step"] = Field(default_factory=list)
    completedMessage: Optional[str] = None

    def find_child(self, *, id: str) -> "Step | None":
        for step in self.progress:
            if isinstance(step, (DefaultStep, RootCauseStep, ChangesStep)) and step.id == id:
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


Step = Union[DefaultStep, RootCauseStep, ChangesStep]


class CodebaseState(BaseModel):
    repo_id: int | None = None
    namespace_id: int | None = None
    repo_external_id: str | None = None
    file_changes: list[FileChange] = []


class AutofixGroupState(BaseModel):
    run_id: int = -1
    steps: list[Step] = Field(default_factory=list)
    status: AutofixStatus = AutofixStatus.PENDING
    codebases: dict[str, CodebaseState] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    last_triggered_at: Optional[
        Annotated[datetime.datetime, Examples(datetime.datetime.now() for _ in gen)]
    ] = None
    updated_at: Optional[
        Annotated[datetime.datetime, Examples(datetime.datetime.now() for _ in gen)]
    ] = None
    completed_at: datetime.datetime | None = None
    signals: list[str] = Field(default_factory=list)


class AutofixStateRequest(BaseModel):
    group_id: int | None = None
    run_id: int | None = None


class AutofixPrIdRequest(BaseModel):
    provider: str
    pr_id: int


class AutofixEvaluationRequest(BaseModel):
    dataset_name: str
    run_name: str
    run_type: Literal["root_cause", "full", "execution"] = "full"
    test: bool = False


class AutofixStateResponse(BaseModel):
    group_id: Optional[int]
    run_id: Optional[int]
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


class AutofixRequestOptions(BaseModel):
    disable_codebase_indexing: bool = False


class AutofixRequest(BaseModel):
    organization_id: Annotated[int, Examples(specialized.unsigned_ints)]
    project_id: Annotated[int, Examples(specialized.unsigned_ints)]
    repos: list[RepoDefinition]
    issue: IssueDetails
    invoking_user: Optional[AutofixUserDetails] = None
    instruction: Optional[str] = Field(default=None, validation_alias="additional_context")

    options: AutofixRequestOptions = Field(default_factory=AutofixRequestOptions)

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


class AutofixRootCauseUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.SELECT_ROOT_CAUSE]
    cause_id: int | None = None
    fix_id: int | None = None
    custom_root_cause: str | None = None


class AutofixCreatePrUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.CREATE_PR]
    repo_external_id: str | None = None
    repo_id: int | None = None  # TODO: Remove this when we won't be breaking LA customers.


class AutofixUpdateRequest(BaseModel):
    run_id: int
    payload: Union[AutofixRootCauseUpdatePayload, AutofixCreatePrUpdatePayload] = Field(
        discriminator="type"
    )


class AutofixContinuation(AutofixGroupState):
    request: AutofixRequest

    def find_step(self, *, id: str) -> Step | None:
        for step in self.steps:
            if step.id == id:
                return step
        return None

    def find_or_add(self, base_step: Step) -> Step:
        existing = self.find_step(id=base_step.id)
        if existing:
            return existing

        base_step = base_step.model_copy()
        base_step.index = len(self.steps)
        self.steps.append(base_step)
        return base_step

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
        root_cause_step = self.find_step(id="root_cause_analysis")
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
