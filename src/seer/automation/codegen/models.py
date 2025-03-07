import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from seer.automation.codebase.models import StaticAnalysisWarning
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import FileChange, IssueDetails, RepoDefinition


class CodegenStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERRORED = "errored"


class RelevantWarningResult(BaseModel):
    warning_id: int
    issue_id: int
    does_fixing_warning_fix_issue: bool
    relevance_probability: float
    reasoning: str
    short_description: str
    short_justification: str
    encoded_location: str


class CodegenState(BaseModel):
    run_id: int = -1
    file_changes: list[FileChange] = Field(default_factory=list)
    relevant_warning_results: list[RelevantWarningResult] = Field(default_factory=list)
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
    request: CodegenBaseRequest

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

    comments: list[Comment]


class CodegenRelevantWarningsRequest(CodegenBaseRequest):
    callback_url: str
    organization_id: int
    warnings: list[StaticAnalysisWarning]
    commit_sha: str
    max_num_associations: int = 10
    max_num_issues_analyzed: int = 10
    should_post_to_overwatch: bool = False


class CodegenRelevantWarningsResponse(CodegenBaseResponse):
    pass


class CodegenRelevantWarningsStateRequest(BaseModel):
    run_id: int


class CodegenRelevantWarningsStateResponse(BaseModel):
    run_id: int
    status: CodegenStatus
    relevant_warning_results: list[RelevantWarningResult]
    triggered_at: datetime.datetime
    updated_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class PrFile(BaseModel):
    # https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28
    filename: str
    patch: str
    status: Literal["added", "removed", "modified", "renamed", "copied", "changed", "unchanged"]
    changes: int


class FilterWarningsRequest(BaseComponentRequest):
    warnings: list[StaticAnalysisWarning]
    target_filenames: list[str]


class FilterWarningsOutput(BaseComponentOutput):
    warnings: list[StaticAnalysisWarning]


class CodeFetchIssuesRequest(BaseComponentRequest):
    organization_id: int
    pr_files: list[PrFile]


class CodeFetchIssuesOutput(BaseComponentOutput):
    filename_to_issues: dict[str, list[IssueDetails]]


class AssociateWarningsWithIssuesRequest(BaseComponentRequest):
    warnings: list[StaticAnalysisWarning]
    filename_to_issues: dict[str, list[IssueDetails]]
    max_num_associations: int


class AssociateWarningsWithIssuesOutput(BaseComponentOutput):
    candidate_associations: list[tuple[StaticAnalysisWarning, IssueDetails]]


class CodeAreIssuesFixableRequest(BaseComponentRequest):
    candidate_issues: list[IssueDetails]
    max_num_issues_analyzed: int


class CodePredictRelevantWarningsRequest(BaseComponentRequest):
    candidate_associations: list[tuple[StaticAnalysisWarning, IssueDetails]]


class CodeAreIssuesFixableOutput(BaseComponentOutput):
    are_fixable: list[bool | None]  # None means the issue was not analyzed


class CodePredictRelevantWarningsOutput(BaseComponentOutput):
    """
    A list of results for all pairs of warnings and issues.
    Includes both relevant and irrelevant warnings.
    """

    relevant_warning_results: list[RelevantWarningResult]


class CodecovTaskRequest(BaseModel):
    data: CodegenUnitTestsRequest | CodegenPrReviewRequest | CodegenRelevantWarningsRequest
    external_owner_id: str
    request_type: Literal["unit-tests", "pr-review", "relevant-warnings"]
