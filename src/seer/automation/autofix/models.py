import datetime
import enum
import uuid
from typing import Annotated, Any, Literal, Optional, Union, cast

from johen import gen
from johen.examples import Examples
from johen.generators import specialized
from pydantic import BaseModel, ConfigDict, Field, field_validator

from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.config import AUTOFIX_HARD_TIME_OUT_MINS, AUTOFIX_UPDATE_TIMEOUT_SECS
from seer.automation.models import FileChange, FilePatch, IssueDetails, RepoDefinition
from seer.automation.summarize.issue import IssueSummary
from seer.db import DbRunMemory


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


class CodeContextRootCauseSelection(BaseModel):
    cause_id: int


RootCauseSelection = Union[CustomRootCauseSelection, CodeContextRootCauseSelection]


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
    # The id is a unique identifier for each individual step.
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # The key is to determine the kind of step, such as root_cause or changes.
    key: str | None = None  # TODO: Make this required when we won't be breaking existing runs.
    title: str
    type: StepType = StepType.DEFAULT

    status: AutofixStatus = AutofixStatus.PROCESSING

    index: int = -1
    progress: list["ProgressItem | Step"] = Field(default_factory=list)
    completedMessage: Optional[str] = None

    queued_user_messages: list[str] = []

    def receive_user_message(self, message: str):
        self.queued_user_messages.append(message)

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

    def model_copy_with_new_id(self):
        new_step = self.model_copy()
        new_step.id = str(uuid.uuid4())
        return new_step

    def ensure_uuid_id(self):
        if self.id and not self.is_valid_uuid(self.id):
            self.key = self.id
            self.id = str(uuid.uuid4())

    @staticmethod
    def is_valid_uuid(uuid_string: str) -> bool:
        try:
            uuid.UUID(uuid_string)
            return True
        except (ValueError, TypeError):
            return False


class DefaultStep(BaseStep):
    type: Literal[StepType.DEFAULT] = StepType.DEFAULT
    insights: list[InsightSharingOutput] = []


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
    status: AutofixStatus = AutofixStatus.PROCESSING
    codebases: dict[str, CodebaseState] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    last_triggered_at: Annotated[
        datetime.datetime, Examples(datetime.datetime.now() for _ in gen)
    ] = Field(default_factory=datetime.datetime.now)
    updated_at: Annotated[datetime.datetime, Examples(datetime.datetime.now() for _ in gen)] = (
        Field(default_factory=datetime.datetime.now)
    )
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
    run_description: Optional[str] = None
    run_type: Literal["root_cause", "full", "execution"] = "full"
    test: bool = False
    run_on_item_id: Optional[str] = None
    random_for_test: bool = False
    n_runs_per_item: int = 1


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
    disable_interactive: bool = False


class AutofixRequest(BaseModel):
    organization_id: Annotated[int, Examples(specialized.unsigned_ints)]
    project_id: Annotated[int, Examples(specialized.unsigned_ints)]
    repos: list[RepoDefinition]
    issue: IssueDetails
    invoking_user: Optional[AutofixUserDetails] = None
    instruction: Optional[str] = Field(default=None, validation_alias="additional_context")
    issue_summary: Optional[IssueSummary] = None

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
    USER_MESSAGE = "user_message"


class AutofixRootCauseUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.SELECT_ROOT_CAUSE]
    cause_id: int | None = None
    custom_root_cause: str | None = None


class AutofixCreatePrUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.CREATE_PR]
    repo_external_id: str | None = None
    repo_id: int | None = None  # TODO: Remove this when we won't be breaking LA customers.


class AutofixUserMessagePayload(BaseModel):
    type: Literal[AutofixUpdateType.USER_MESSAGE]
    text: str


class AutofixUpdateRequest(BaseModel):
    run_id: int
    payload: Union[
        AutofixRootCauseUpdatePayload, AutofixCreatePrUpdatePayload, AutofixUserMessagePayload
    ] = Field(discriminator="type")


class AutofixContinuation(AutofixGroupState):
    request: AutofixRequest

    def get_step_description(self) -> str:
        if not self.steps:
            return ""
        step = self.steps[-1]
        if step.type == StepType.DEFAULT and step.key == "root_cause_analysis_processing":
            return "figuring out what is causing the issue (not thinking about solutions yet)"
        elif step.type == StepType.DEFAULT and step.key == "plan":
            return "coming up with a fix for the issue"
        elif step.type == StepType.ROOT_CAUSE_ANALYSIS:
            return "selecting the final root cause"
        elif step.type == StepType.CHANGES:
            return "writing the code changes to fix the issue"
        else:
            return ""

    def find_step(self, *, id: str | None = None, key: str | None = None) -> Step | None:
        for step in self.steps[::-1]:
            if step.id == id:
                return step
            if key is not None and step.key == key:
                return step
        return None

    def find_or_add(self, base_step: Step) -> Step:
        existing = self.find_step(key=base_step.key)
        if existing:
            return existing

        base_step = base_step.model_copy_with_new_id()
        return self.add_step(base_step)

    def add_step(self, step: Step):
        step.index = len(self.steps)
        self.steps.append(step)
        return step

    def make_step_latest(self, step: Step):
        if step in self.steps:
            self.steps.remove(step)
            self.steps.append(step)

    def mark_running_steps_completed(self):
        for step in self.steps:
            if step.status == AutofixStatus.PROCESSING:
                step.status = AutofixStatus.COMPLETED

    def mark_running_steps_errored(self):
        did_mark = False
        for step in self.steps:
            if step.status == AutofixStatus.PROCESSING:
                step.status = AutofixStatus.ERROR
                did_mark = True
                for substep in step.progress:
                    if isinstance(substep, (DefaultStep, RootCauseStep, ChangesStep)):
                        if substep.status == AutofixStatus.PROCESSING:
                            substep.status = AutofixStatus.ERROR

        return did_mark

    def set_last_step_completed_message(self, message: str):
        if self.steps:
            self.steps[-1].completedMessage = message

    def get_selected_root_cause_and_fix(self) -> RootCauseAnalysisItem | str | None:
        root_cause_step = self.find_step(key="root_cause_analysis")
        if root_cause_step and isinstance(root_cause_step, RootCauseStep):
            if root_cause_step.selection:
                if isinstance(root_cause_step.selection, CodeContextRootCauseSelection):
                    cause = next(
                        cause
                        for cause in root_cause_step.causes
                        if cause.id == root_cause_step.selection.cause_id
                    )
                    return cause
                elif isinstance(root_cause_step.selection, CustomRootCauseSelection):
                    return root_cause_step.selection.custom_root_cause
        return None

    def mark_triggered(self):
        self.last_triggered_at = datetime.datetime.now()

    def mark_updated(self):
        self.updated_at = datetime.datetime.now()

    def delete_steps_after(self, step: Step, include_current: bool = False):
        found_index = next((i for i, s in enumerate(self.steps) if s.id == step.id), -1)
        if found_index != -1:
            self.steps = self.steps[: found_index + (0 if include_current else 1)]

    def clear_file_changes(self):
        for key, codebase in self.codebases.items():
            codebase.file_changes = []
            self.codebases[key] = codebase

    def get_all_insights(self):
        insights = []
        step = self.steps[-1]
        if step.status != AutofixStatus.ERROR and isinstance(step, DefaultStep):
            for insight in cast(DefaultStep, step).insights:
                insights.append(insight.insight)
        return insights

    @property
    def is_running(self):
        return self.status == AutofixStatus.PROCESSING

    @property
    def has_timed_out(self) -> bool:
        if self.is_running and self.last_triggered_at:
            now = datetime.datetime.now()

            # If it's still processing and there hasn't been an update in 90 seconds, we consider it timed out.
            if (
                self.updated_at
                and self.updated_at + datetime.timedelta(seconds=AUTOFIX_UPDATE_TIMEOUT_SECS) <= now
            ):
                return True

            # If an autofix run has been running for more than 10 minutes, we consider it timed out.
            return (
                self.last_triggered_at + datetime.timedelta(minutes=AUTOFIX_HARD_TIME_OUT_MINS)
                < now
            )
        return False


class AutofixRunMemory(BaseModel):
    run_id: int
    memory: dict[str, list[Message]] = Field(default_factory=dict)

    def to_db_model(self) -> DbRunMemory:
        return DbRunMemory(run_id=self.run_id, value=self.model_dump(mode="json"))

    @classmethod
    def from_db_model(cls, model: DbRunMemory) -> "AutofixRunMemory":
        return cls.model_validate(model.value)
