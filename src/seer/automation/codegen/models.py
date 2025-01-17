import datetime
from enum import Enum
from typing import List, Literal, Union

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


class CodegenBaseRequest(BaseModel):
    repo: RepoDefinition
    pr_id: int  # The PR number


class CodegenUnitTestsRequest(CodegenBaseRequest):
    pass


class CodegenPrReviewRequest(CodegenBaseRequest):
    pass


class CodegenContinuation(CodegenState):
    request: Union[CodegenUnitTestsRequest, CodegenPrReviewRequest]

    def mark_triggered(self):
        self.last_triggered_at = datetime.datetime.now()

    def mark_updated(self):
        self.updated_at = datetime.datetime.now()


class CodeUnitTestRequest(BaseComponentRequest):
    diff: str
    codecov_client_params: dict = Field(default_factory=dict)


class CodegenBaseResponse(BaseModel):
    run_id: int


class CodegenPrReviewResponse(CodegenBaseResponse):
    pass


class CodegenUnitTestsResponse(CodegenBaseResponse):
    pass


class CodegenUnitTestsStateRequest(BaseModel):
    run_id: int


class CodegenUnitTestsStateResponse(BaseModel):
    run_id: int
    status: CodegenStatus
    changes: list[FileChange]
    triggered_at: datetime.datetime
    updated_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class CodegenPrReviewStateRequest(BaseModel):
    run_id: int


class CodegenPrReviewStateResponse(BaseModel):
    run_id: int
    status: CodegenStatus
    changes: list[FileChange]
    triggered_at: datetime.datetime
    updated_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class CodePrReviewRequest(BaseComponentRequest):
    diff: str


class CodePrReviewOutput(BaseComponentOutput):
    class Comment(BaseModel):
        path: str
        line: int
        body: str
        start_line: int

    comments: List[Comment]


class CodecovTaskRequest(BaseModel):
    data: CodegenPrReviewRequest | CodegenUnitTestsRequest
    external_owner_id: str
    request_type: Literal["unit-tests", "pr-review"]
