import datetime
import enum
import uuid
from typing import Annotated, Any, Literal, Optional, Set, Union

from johen import gen
from johen.examples import Examples
from johen.generators import specialized
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.components.solution.models import SolutionTimelineEvent
from seer.automation.autofix.config import AUTOFIX_HARD_TIME_OUT_MINS, AUTOFIX_UPDATE_TIMEOUT_SECS
from seer.automation.models import (
    FileChange,
    FilePatch,
    IssueDetails,
    Line,
    Profile,
    RepoDefinition,
    TraceTree,
)
from seer.automation.summarize.issue import IssueSummary
from seer.automation.utils import make_kill_signal
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
    WAITING_FOR_USER_RESPONSE = "WAITING_FOR_USER_RESPONSE"

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


class AutofixUpdateEndpointResponse(BaseModel):
    run_id: int
    status: Literal["success", "error"] = "success"
    message: str | None = None


class CustomRootCauseSelection(BaseModel):
    custom_root_cause: str


class CodeContextRootCauseSelection(BaseModel):
    cause_id: int
    instruction: str | None = None


RootCauseSelection = Union[CustomRootCauseSelection, CodeContextRootCauseSelection]


class CommittedPullRequestDetails(BaseModel):
    pr_number: int
    pr_url: str
    pr_id: Optional[int] = None


class CodebaseChange(BaseModel):
    repo_external_id: str | None = None
    repo_name: str
    title: str
    description: str
    diff: list[FilePatch] = []
    diff_str: Optional[str] = None
    draft_branch_name: str | None = None
    branch_name: str | None = None
    pull_request: Optional[CommittedPullRequestDetails] = None


class CommentThread(BaseModel):
    id: str
    messages: list[Message] = []
    is_completed: bool = False
    selected_text: str | None = None


class StepType(str, enum.Enum):
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    SOLUTION = "solution"
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
    output_stream: str | None = None
    active_comment_thread: CommentThread | None = None  # user-initiated comment thread
    agent_comment_thread: CommentThread | None = None  # Autofix-initiated comment thread
    output_confidence_score: float | None = None  # confidence in the step's output
    proceed_confidence_score: float | None = None  # confidence in proceeding to the next step

    def receive_user_message(self, message: str):
        self.queued_user_messages.append(message)

    def receive_output_stream(self, stream_chunk: str):
        if self.output_stream is None:
            self.output_stream = ""
        self.output_stream += stream_chunk

    def clear_output_stream(self):
        self.output_stream = None

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
    initial_memory_length: int = 1

    def get_all_insights(self, exclude_user_messages: bool = False):
        insights = []
        if self.status != AutofixStatus.ERROR and isinstance(self, DefaultStep):
            for insight in self.insights:
                if not exclude_user_messages or insight.justification != "USER":
                    insights.append(insight.insight)
        return insights


class RootCauseStep(BaseStep):
    type: Literal[StepType.ROOT_CAUSE_ANALYSIS] = StepType.ROOT_CAUSE_ANALYSIS

    causes: list[RootCauseAnalysisItem] = []
    selection: RootCauseSelection | None = None
    termination_reason: str | None = None


class SolutionStep(BaseStep):
    type: Literal[StepType.SOLUTION] = StepType.SOLUTION

    solution: list[SolutionTimelineEvent] = []
    description: str | None = None
    custom_solution: str | None = None
    solution_selected: bool = False
    selected_mode: Literal["all", "fix", "test"] | None = None


class ChangesStep(BaseStep):
    type: Literal[StepType.CHANGES] = StepType.CHANGES

    changes: list[CodebaseChange]


Step = Union[DefaultStep, RootCauseStep, ChangesStep, SolutionStep]


class CodebaseState(BaseModel):
    repo_external_id: str | None = None
    file_changes: list[FileChange] = []

    is_readable: bool | None = None
    is_writeable: bool | None = None


class AutofixFeedback(BaseModel):
    root_cause_thumbs_up: bool | None = None
    root_cause_thumbs_down: bool | None = None


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
    feedback: AutofixFeedback | None = None


class AutofixStateRequest(BaseModel):
    group_id: int | None = None
    run_id: int | None = None
    check_repo_access: bool = False


class AutofixPrIdRequest(BaseModel):
    provider: str
    pr_id: int


class AutofixEvaluationRequest(BaseModel):
    dataset_name: str
    run_name: str
    run_description: Optional[str] = None
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
    disable_codebase_indexing: bool = False
    comment_on_pr_with_url: str | None = None
    disable_interactivity: bool = False


class AutofixRequest(BaseModel):
    organization_id: Annotated[int, Examples(specialized.unsigned_ints)]
    project_id: Annotated[int, Examples(specialized.unsigned_ints)]
    repos: list[RepoDefinition]
    issue: IssueDetails
    invoking_user: AutofixUserDetails | None = None
    instruction: str | None = Field(default=None, validation_alias="additional_context")
    issue_summary: IssueSummary | None = None
    profile: Profile | None = None
    trace_tree: TraceTree | None = None

    options: AutofixRequestOptions = Field(default_factory=AutofixRequestOptions)

    @field_validator("profile", mode="before")
    @classmethod
    def extract_relevant_functions(
        cls, profile: Profile | None, info: ValidationInfo
    ) -> Profile | None:
        if profile is not None and "issue" in info.data:
            issue = info.data["issue"]
            if not isinstance(issue, IssueDetails):
                return profile

            relevant_functions: Set[str] = set()

            # Extract functions from exceptions
            for event in issue.events:
                for entry in event.get("entries", []):
                    if entry.get("type") == "exception":
                        for exception in entry.get("data", {}).get("values", []):
                            if "stacktrace" in exception:
                                for frame in exception["stacktrace"].get("frames", []):
                                    if frame.get("function") and frame.get("in_app", False):
                                        relevant_functions.add(frame["function"])

            if relevant_functions:
                profile.relevant_functions = relevant_functions

        return profile

    @field_validator("issue_summary", mode="before")
    @classmethod
    def validate_issue_summary(cls, value):
        if value is None:
            return None
        try:
            return IssueSummary.model_validate(value)
        except Exception:
            return None

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
    SELECT_SOLUTION = "select_solution"
    CREATE_PR = "create_pr"
    CREATE_BRANCH = "create_branch"
    USER_MESSAGE = "user_message"
    RESTART_FROM_POINT_WITH_FEEDBACK = "restart_from_point_with_feedback"
    UPDATE_CODE_CHANGE = "update_code_change"
    COMMENT_THREAD = "comment_thread"
    RESOLVE_COMMENT_THREAD = "resolve_comment_thread"
    FEEDBACK = "feedback"


class AutofixRootCauseUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.SELECT_ROOT_CAUSE] = AutofixUpdateType.SELECT_ROOT_CAUSE
    cause_id: int | None = None
    custom_root_cause: str | None = None
    instruction: str | None = None


class AutofixSolutionUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.SELECT_SOLUTION] = AutofixUpdateType.SELECT_SOLUTION
    custom_solution: str | None = None
    solution_selected: bool = False
    mode: Literal["all", "fix", "test"] = "fix"
    solution: list[SolutionTimelineEvent] | None = None


class AutofixCreatePrUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.CREATE_PR] = AutofixUpdateType.CREATE_PR
    repo_external_id: str | None = None
    make_pr: bool = True


class AutofixCreateBranchUpdatePayload(BaseModel):
    type: Literal[AutofixUpdateType.CREATE_BRANCH] = AutofixUpdateType.CREATE_BRANCH
    repo_external_id: str | None = None
    make_pr: bool = False


class AutofixUserMessagePayload(BaseModel):
    type: Literal[AutofixUpdateType.USER_MESSAGE]
    text: str


class AutofixRestartFromPointPayload(BaseModel):
    type: Literal[AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK]
    message: str
    step_index: int
    retain_insight_card_index: int | None = None
    add_to_insights: bool = True


class AutofixUpdateCodeChangePayload(BaseModel):
    type: Literal[AutofixUpdateType.UPDATE_CODE_CHANGE]
    hunk_index: int
    lines: list[Line]
    file_path: str
    repo_external_id: str | None = Field(default=None, alias="repo_id")

    model_config = ConfigDict(populate_by_name=True)  # Allows both field name and alias


class AutofixCommentThreadPayload(BaseModel):
    type: Literal[AutofixUpdateType.COMMENT_THREAD]
    thread_id: str
    selected_text: str | None = None
    message: str
    step_index: int
    retain_insight_card_index: int | None = None
    is_agent_comment: bool = False


class AutofixResolveCommentThreadPayload(BaseModel):
    type: Literal[AutofixUpdateType.RESOLVE_COMMENT_THREAD]
    thread_id: str
    step_index: int
    is_agent_comment: bool = False


class AutofixFeedbackPayload(BaseModel):
    type: Literal[AutofixUpdateType.FEEDBACK]
    action: Literal["root_cause_thumbs_up", "root_cause_thumbs_down"]


class AutofixUpdateRequest(BaseModel):
    run_id: int
    payload: Union[
        AutofixRootCauseUpdatePayload,
        AutofixSolutionUpdatePayload,
        AutofixCreatePrUpdatePayload,
        AutofixCreateBranchUpdatePayload,
        AutofixUserMessagePayload,
        AutofixRestartFromPointPayload,
        AutofixUpdateCodeChangePayload,
        AutofixCommentThreadPayload,
        AutofixResolveCommentThreadPayload,
        AutofixFeedbackPayload,
    ] = Field(discriminator="type")


class AutofixContinuation(AutofixGroupState):
    request: AutofixRequest

    @property
    def readable_repos(self) -> list[RepoDefinition]:
        return [
            repo
            for repo in self.request.repos
            if self.codebases[repo.external_id].is_readable is True
            or self.codebases[repo.external_id].is_readable
            is None  # TODO: Remove this once we don't need backwards compatibility
        ]

    @property
    def unreadable_repos(self) -> list[RepoDefinition]:
        return [
            repo
            for repo in self.request.repos
            if self.codebases[repo.external_id].is_readable is False
        ]

    def kill_all_processing_steps(self):
        for step in self.steps:
            if step.status == AutofixStatus.PROCESSING:
                self.signals.append(make_kill_signal())

    def find_step(
        self, *, id: str | None = None, key: str | None = None, index: int | None = None
    ) -> Step | None:
        if index is not None and 0 <= index < len(self.steps):
            return self.steps[index]
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

    def find_last_step_waiting_for_response(self) -> Step | None:
        for step in self.steps[::-1]:
            if step.status == AutofixStatus.WAITING_FOR_USER_RESPONSE:
                return step
        return None

    def add_step(self, step: Step):
        step.index = len(self.steps)
        self.steps.append(step)
        return step

    def delete_all_steps_after_index(self, index: int):
        if 0 <= index < len(self.steps):
            self.steps = self.steps[: index + 1]

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

    @property
    def root_cause_step(self) -> RootCauseStep | None:
        root_cause_step = self.find_step(key="root_cause_analysis")
        return (
            root_cause_step
            if root_cause_step and isinstance(root_cause_step, RootCauseStep)
            else None
        )

    @property
    def solution_step(self) -> SolutionStep | None:
        solution_step = self.find_step(key="solution")

        return solution_step if solution_step and isinstance(solution_step, SolutionStep) else None

    @property
    def changes_step(self) -> ChangesStep | None:
        changes_step = self.find_step(key="changes")
        return changes_step if changes_step and isinstance(changes_step, ChangesStep) else None

    def get_selected_root_cause(
        self,
    ) -> tuple[RootCauseAnalysisItem | str | None, str | None]:
        root_cause_step = self.find_step(key="root_cause_analysis")
        if root_cause_step and isinstance(root_cause_step, RootCauseStep):
            if root_cause_step.selection:
                if isinstance(root_cause_step.selection, CodeContextRootCauseSelection):
                    cause = next(
                        cause
                        for cause in root_cause_step.causes
                        if cause.id == root_cause_step.selection.cause_id
                    )
                    return cause, root_cause_step.selection.instruction
                elif isinstance(root_cause_step.selection, CustomRootCauseSelection):
                    return root_cause_step.selection.custom_root_cause, None
        return None, None

    def get_selected_solution(
        self,
    ) -> tuple[list[SolutionTimelineEvent] | str | None, Literal["all", "fix", "test"] | None]:
        solution_step = self.find_step(key="solution")
        if solution_step and isinstance(solution_step, SolutionStep):
            if solution_step.solution_selected:
                if solution_step.custom_solution:
                    return solution_step.custom_solution, solution_step.selected_mode
                else:
                    return solution_step.solution, solution_step.selected_mode
        return None, None

    def mark_triggered(self):
        self.last_triggered_at = datetime.datetime.now()

    def mark_updated(self):
        self.updated_at = datetime.datetime.now()

    def delete_steps_after(self, step: Step, include_current: bool = False):
        found_index = next((i for i, s in enumerate(self.steps) if s.id == step.id), -1)
        if found_index != -1:
            self.steps = self.steps[: found_index + (0 if include_current else 1)]

    def clear_file_changes(self):
        for codebase in self.codebases.values():
            codebase.file_changes = []

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
