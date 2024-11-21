import datetime
from enum import Enum

from pydantic import BaseModel, Field

from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import FileChange, RepoDefinition


class CodegenStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERRORED = "errored"


class CodegenState(BaseModel):
    run_id: int = -1
    file_changes: list[FileChange] = Field(default_factory=list)
    status: CodegenStatus = CodegenStatus.PENDING
    last_triggered_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    completed_at: datetime.datetime | None = None
    signals: list[str] = Field(default_factory=list)


class CodeUnitTestOutput(BaseComponentOutput):
    diffs: list[FileChange]


class CodegenUnitTestsRequest(BaseModel):
    repo: RepoDefinition
    pr_id: int  # The PR number


class CodegenPrReviewRequest(BaseModel):
    repo: RepoDefinition
    pr_id: int  # The PR number


class CodegenContinuation(CodegenState):
    request: CodegenUnitTestsRequest

    def mark_triggered(self):
        self.last_triggered_at = datetime.datetime.now()

    def mark_updated(self):
        self.updated_at = datetime.datetime.now()


class CodeUnitTestRequest(BaseComponentRequest):
    diff: str
    codecov_client_params: dict = Field(default_factory=dict)


class CodegenUnitTestsResponse(BaseModel):
    run_id: int


class CodegenPrReviewResponse(BaseModel):
    run_id: int


class CodegenUnitTestsStateRequest(BaseModel):
    run_id: int


class CodegenUnitTestsStateResponse(BaseModel):
    run_id: int
    status: CodegenStatus
    changes: list[FileChange]
    triggered_at: datetime.datetime
    updated_at: datetime.datetime
    completed_at: datetime.datetime | None = None
