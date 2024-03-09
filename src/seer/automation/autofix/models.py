import datetime
import enum
from typing import Annotated, Any, Literal, Optional

import sentry_sdk
from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from pydantic.alias_generators import to_camel, to_snake
from typing_extensions import NotRequired, TypedDict

from seer import generator
from seer.automation.agent.models import Usage
from seer.automation.models import FilePatch
from seer.generator import Examples


class FileChangeError(Exception):
    pass


class ProgressType(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    NEED_MORE_INFORMATION = "NEED_MORE_INFORMATION"
    USER_RESPONSE = "USER_RESPONSE"


class ProgressItem(BaseModel):
    timestamp: str
    message: str
    type: ProgressType
    data: Any = None


class AutofixStatus(enum.Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    CANCELLED = "CANCELLED"

    @classmethod
    def terminal(cls) -> "frozenset[AutofixStatus]":
        return frozenset((cls.COMPLETED, cls.ERROR, cls.CANCELLED))


class PlanStep(BaseModel):
    id: int
    title: str
    text: str


class PlanningOutput(BaseModel):
    title: str
    description: str
    steps: list[PlanStep]


class ProblemDiscoveryOutput(BaseModel):
    description: str
    reasoning: str
    actionability_score: float


class ProblemDiscoveryResult(BaseModel):
    status: Literal["CONTINUE", "CANCELLED"]
    description: str
    reasoning: str


class ProblemDiscoveryRequest(BaseModel):
    message: str
    previous_output: ProblemDiscoveryOutput


class PlanningInput(BaseModel):
    message: Optional[str] = None
    previous_output: Optional[PlanningOutput] = None
    problem: Optional[ProblemDiscoveryOutput] = None


class StacktraceFrame(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda k: AliasChoices(to_camel(k), to_snake(k)),
            serialization_alias=to_camel,
        )
    )

    function: Annotated[str, Examples(generator.ascii_words)]
    filename: Annotated[str, Examples(generator.file_names)]
    abs_path: Annotated[str, Examples(generator.file_paths)]
    line_no: Optional[int]
    col_no: Optional[int]
    context: list[tuple[int, str]]
    repo_name: Optional[str] = None
    repo_id: Optional[int] = None
    in_app: bool = False


class Stacktrace(BaseModel):
    frames: list[StacktraceFrame]

    def to_str(self, max_frames: int = 16):
        stack_str = ""
        for frame in reversed(self.frames[-max_frames:]):
            col_no_str = f", column {frame.col_no}" if frame.col_no is not None else ""
            repo_str = f" in repo {frame.repo_name}" if frame.repo_name else ""
            line_no_str = (
                f"[Line {frame.line_no}{col_no_str}]"
                if frame.line_no is not None
                else "[Line: Unknown]"
            )
            stack_str += f" {frame.function} in file {frame.filename}{repo_str} {line_no_str} ({'In app' if frame.in_app else 'Not in app'})\n"
            for ctx in frame.context:
                is_suspect_line = ctx[0] == frame.line_no
                stack_str += f"{ctx[1]}{'  <-- SUSPECT LINE' if is_suspect_line else ''}\n"
            stack_str += "------\n"
        return stack_str


class SentryFrame(TypedDict):
    absPath: Optional[str]
    colNo: Optional[int]
    context: list[tuple[int, str]]
    filename: Optional[str]
    function: Optional[str]
    inApp: bool
    instructionAddr: Optional[str]
    lineNo: Optional[int]
    module: Optional[str]
    package: Optional[str]
    platform: Optional[str]
    rawFunction: Optional[str]
    symbol: Optional[str]
    symbolAddr: Optional[str]
    trust: Optional[Any]
    vars: Optional[dict[str, Any]]
    addrMode: NotRequired[Optional[str]]
    isPrefix: NotRequired[bool]
    isSentinel: NotRequired[bool]
    lock: NotRequired[Optional[Any]]
    map: NotRequired[Optional[str]]
    mapUrl: NotRequired[Optional[str]]
    minGroupingLevel: NotRequired[int]
    origAbsPath: NotRequired[Optional[str]]
    sourceLink: NotRequired[Optional[str]]
    symbolicatorStatus: NotRequired[Optional[Any]]


class SentryStacktrace(TypedDict):
    frames: list[SentryFrame]


class SentryEventEntryDataValue(TypedDict):
    stacktrace: SentryStacktrace


class SentryExceptionEventData(TypedDict):
    values: list[SentryEventEntryDataValue]


class SentryExceptionEntry(BaseModel):
    type: Literal["exception"]
    data: SentryExceptionEventData


class SentryEvent(BaseModel):
    entries: list[dict]

    def get_stacktrace(self) -> Stacktrace | None:
        exception_entry: SentryExceptionEntry | None = None
        for value in self.entries:
            if value.get("type") != "exception":
                continue

            try:
                exception_entry = SentryExceptionEntry.model_validate(value)
                break
            except ValidationError:
                sentry_sdk.capture_exception()
                continue

        if exception_entry is None:
            return None

        values = exception_entry.data["values"]
        if not values:
            return None

        for value in values:
            stack_trace = value["stacktrace"]
            frames: list[StacktraceFrame] = []

            for frame in stack_trace["frames"]:
                try:
                    frames.append(StacktraceFrame.model_validate(frame))
                except ValidationError as e:
                    sentry_sdk.capture_exception()
                    continue

            if not frames:
                continue

            return Stacktrace(frames=frames)
        return None


class IssueDetails(BaseModel):
    id: int
    title: str
    short_id: Optional[str] = None
    events: list[SentryEvent]


class RepoDefinition(BaseModel):
    provider: Annotated[str, Examples(("github", "integrations:github"))]
    owner: str
    name: str

    @property
    def full_name(self):
        return f"{self.owner}/{self.name}"

    @field_validator("provider", mode="after")
    @classmethod
    def validate_provider(cls, provider: str):
        cleaned_provider = provider
        if provider.startswith("integrations:"):
            cleaned_provider = provider.split(":")[1]

        if cleaned_provider != "github":
            raise ValueError(f"Provider {cleaned_provider} is not supported.")

        return cleaned_provider

    def __hash__(self):
        return hash((self.provider, self.owner, self.name))


class AutofixUserDetails(BaseModel):
    id: int
    display_name: str


class AutofixRequest(BaseModel):
    organization_id: int
    project_id: int
    repos: list[RepoDefinition]
    issue: IssueDetails
    invoking_user: Optional[AutofixUserDetails] = None

    base_commit_sha: Optional[str] = None
    additional_context: Optional[str] = None
    timeout_secs: Optional[int] = None
    last_updated: Optional[datetime.datetime] = None

    @property
    def process_request_name(self) -> str:
        return f"{self.organization_id}:{self.issue.id}"

    @property
    def has_timed_out(self, now: datetime.datetime | None = None) -> bool:
        if self.timeout_secs and self.last_updated:
            if now is None:
                now = datetime.datetime.now()
            return self.last_updated + datetime.timedelta(seconds=self.timeout_secs) < now
        return False

    @field_validator("repos", mode="after")
    @classmethod
    def validate_repo_duplicates(cls, repos):
        if isinstance(repos, list):
            # Check for duplicates by comparing lengths after converting to a set
            if len(set(repos)) != len(repos):
                raise ValueError("Duplicate repos detected in the request.")
            return repos

        raise ValueError("Not a list of repos.")


class AutofixOutput(BaseModel):
    title: str
    description: str
    pr_url: str
    pr_number: int
    repo_name: str
    diff: Optional[list[FilePatch]] = []
    usage: Usage


class AutofixEndpointResponse(BaseModel):
    started: bool


class PullRequestResult(BaseModel):
    pr_number: int
    pr_url: str
    repo: RepoDefinition
    diff: list[FilePatch]


class Step(BaseModel):
    id: str
    title: str

    status: AutofixStatus = AutofixStatus.PENDING

    index: int = -1
    progress: list["ProgressItem | Step"] = Field(default_factory=list)
    completedMessage: Optional[str] = None

    def find_child(self, *, id: str) -> "Step | None":
        for step in self.progress:
            if isinstance(step, Step) and step.id == id:
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


class AutofixContinuation(BaseModel):
    request: AutofixRequest
    steps: list[Step] = Field(default_factory=list)
    status: AutofixStatus = Field(default=AutofixStatus.PENDING)
    fix: AutofixOutput | None = None
    completedAt: datetime.datetime | None = None

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
