import datetime
import enum
import hashlib
from typing import Annotated, Any, Literal, Optional

from johen import gen
from johen.examples import Examples
from johen.generators import specialized
from pydantic import BaseModel, ConfigDict, Field, field_validator

from seer.automation.agent.models import Usage
from seer.automation.models import FilePatch, IssueDetails, RepoDefinition


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


class ProblemDiscoveryResult(BaseModel):
    status: Literal["CONTINUE", "CANCELLED"]
    description: str
    reasoning: str


class AutofixUserDetails(BaseModel):
    id: Annotated[int, Examples(specialized.unsigned_ints)]
    display_name: str


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
    timeout_secs: Optional[Annotated[int, Examples((60 * 5,))]] = None
    last_updated: Optional[
        Annotated[datetime.datetime, Examples(datetime.datetime.now() for _ in gen)]
    ] = None

    @property
    def process_request_name(self) -> str:
        return f"autofix:{self.organization_id}:{self.issue.id}"

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

    model_config = ConfigDict(
        populate_by_name=True,
    )


class AutofixOutput(BaseModel):
    title: str
    description: str
    pr_url: str
    pr_number: int
    repo_name: str
    diff: Optional[list[FilePatch]] = []
    diff_str: Optional[str] = None
    usage: Usage


class AutofixEndpointResponse(BaseModel):
    started: bool


class PullRequestResult(BaseModel):
    pr_number: int
    pr_url: str
    repo: RepoDefinition
    diff: list[FilePatch]
    diff_str: Optional[str] = None


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


class AutofixGroupState(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    status: AutofixStatus = AutofixStatus.PENDING
    fix: AutofixOutput | None = None
    completedAt: datetime.datetime | None = None
    usage: Usage = Field(default_factory=Usage)


class AutofixCompleteArgs(BaseModel):
    issue_id: int
    status: AutofixStatus
    steps: list[Step]
    fix: AutofixOutput | None


class AutofixStepUpdateArgs(BaseModel):
    issue_id: int
    status: AutofixStatus
    steps: list[Step]


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
