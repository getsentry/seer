import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from seer.automation.agent.models import Message
from seer.automation.codebase.models import Location, StaticAnalysisWarning
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import FileChange, IssueDetails, RepoDefinition
from seer.db import DbRunMemory


class CodegenStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERRORED = "errored"


class RelevantWarningResult(BaseModel):
    warning_id: int | None
    issue_id: int | None
    does_fixing_warning_fix_issue: bool
    relevance_probability: float
    reasoning: str
    short_description: str
    short_justification: str
    encoded_location: str


class StaticAnalysisSuggestion(BaseModel):
    path: str = Field(description="The path to the file that contains the suggestion.")
    line: int = Field(description="The line number of the suggestion.")
    short_description: str = Field(
        description="A short, fluff-free, information-dense description of the problem. Max 30 words."
    )
    justification: str = Field(
        description="A short, fluff-free, information-dense summary of your analysis for why this is a problem. This justification should be at most 15 words."
    )
    related_warning_id: str | None = Field(
        default=None,
        description="If this suggestion is based on a warning, include the warning id here. Else use null.",
    )
    related_issue_id: str | None = Field(
        default=None,
        description="If this suggestion is based on an issue, include the issue id here. Else use null.",
    )
    severity_score: float = Field(
        description="From 0 to 1 how serious is this potential bug? 1 being 'guaranteed exception will happen and not be caught by the code'."
    )
    confidence_score: float = Field(
        description="From 0 to 1 how confident are you that this is a bug? 1 being 'I am 100% confident that this is a bug'. This should be based on the amount of evidence you had to reach your conclusion."
    )
    missing_evidence: list[str] = Field(
        description="A short list of evidence that you did NOT have but would increase your confidence score. At most 5 items. Be very specific."
    )

    def to_overwatch_format(self) -> RelevantWarningResult:
        """
        Convert a StaticAnalysisSuggestion to a RelevantWarningResult.
        This is a temporary format to post to Overwatch, because Overwatch expects a list of
        RelevantWarningResult objects at this time
        TODO: update Overwatch and then remove this method
        """
        return RelevantWarningResult(
            warning_id=int(self.related_warning_id) if self.related_warning_id else None,
            issue_id=int(self.related_issue_id) if self.related_issue_id else None,
            # Let's pretend our suggestions are important
            does_fixing_warning_fix_issue=True,
            # Combining both metrics means we will only surface suggestions with high confidence
            # and severity.
            relevance_probability=self.confidence_score * self.severity_score,
            reasoning=self.justification,
            short_justification=self.justification,
            short_description=self.short_description,
            encoded_location=Location(
                filename=self.path, start_line=str(self.line), end_line=str(self.line)
            ).encode(),
        )


class CodegenState(BaseModel):
    run_id: int = -1
    file_changes: list[FileChange] = Field(default_factory=list)
    static_analysis_suggestions: list[StaticAnalysisSuggestion] = Field(default_factory=list)
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
    codecov_status: dict[str, str] | None = None


class CodegenUnitTestsRequest(CodegenBaseRequest):
    pass


class CodegenPrReviewRequest(CodegenBaseRequest):
    pass


class CodegenPrClosedRequest(CodegenBaseRequest):
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


class CodegenPrClosedResponse(CodegenBaseResponse):
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
        suggestion: str | None = None

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
    filename: str
    patch: str
    status: Literal["added", "removed", "modified", "renamed", "copied", "changed", "unchanged"]
    changes: int
    sha: str


class FilterWarningsRequest(BaseComponentRequest):
    warnings: list[StaticAnalysisWarning]
    pr_files: list[PrFile]


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


class CodePredictStaticAnalysisSuggestionsRequest(BaseComponentRequest):
    warnings: list[StaticAnalysisWarning]
    fixable_issues: list[IssueDetails]
    pr_files: list[PrFile]


class CodePredictStaticAnalysisSuggestionsOutput(BaseComponentOutput):
    suggestions: list[StaticAnalysisSuggestion]


class CodePredictRelevantWarningsOutput(BaseComponentOutput):
    """
    A list of results for all pairs of warnings and issues.
    Includes both relevant and irrelevant warnings.
    """

    relevant_warning_results: list[RelevantWarningResult]


class CodecovTaskRequest(BaseModel):
    data: (
        CodegenUnitTestsRequest
        | CodegenPrReviewRequest
        | CodegenRelevantWarningsRequest
        | CodegenPrClosedRequest
    )
    external_owner_id: str
    request_type: Literal[
        "unit-tests", "pr-review", "relevant-warnings", "pr-closed", "retry-unit-tests"
    ]


class UnitTestRunMemory(BaseModel):
    run_id: int
    memory: dict[str, list[Message]] = Field(default_factory=dict)

    def to_db_model(self) -> DbRunMemory:
        return DbRunMemory(run_id=self.run_id, value=self.model_dump(mode="json"))

    @classmethod
    def from_db_model(cls, model: DbRunMemory) -> "UnitTestRunMemory":
        return cls.model_validate(model.value)
