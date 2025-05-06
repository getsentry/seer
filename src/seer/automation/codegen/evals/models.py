from pydantic import BaseModel, field_validator
from pydantic.fields import Field

from seer.automation.codebase.models import PrFile, StaticAnalysisWarning
from seer.automation.codegen.models import CodegenRelevantWarningsRequest
from seer.automation.models import IssueDetails


class RepoInfo(BaseModel):
    provider: str
    owner: str
    name: str
    external_id: str


class EvalItemInput(BaseModel):
    """
    An item in the evaluation dataset.
    """

    repo: RepoInfo
    pr_id: int
    organization_id: int
    commit_sha: str
    warnings: list[StaticAnalysisWarning]
    pr_files: list[PrFile] | None = None
    issues: list[IssueDetails] | None = None

    @field_validator("issues", mode="after")
    def set_issues_to_empty_list_if_none(cls, v):
        if v is None:
            return []
        return v

    def get_request(self) -> CodegenRelevantWarningsRequest:
        return CodegenRelevantWarningsRequest(
            repo=self.repo,
            pr_id=self.pr_id,
            organization_id=self.organization_id,
            commit_sha=self.commit_sha,
            callback_url="",
            warnings=self.warnings,
        )


class EvalItemOutput(BaseModel):
    repos: list[str]
    description: str
    encoded_location: str


class ModelEvaluationOutput(BaseModel):
    suggestion_match_idx: int = Field(
        description="The index of the suggestion that matches the bug"
    )
    actual_bug_idx: int = Field(
        description="The index of the actual bug in the list of expected issues"
    )
    match_score: float = Field(
        description="The score for the match between the suggestion and the actual bug, from 0 to 1"
    )
    reasoning: str = Field(description="A short explanation of the match score")


class ModelEvaluationOutputList(BaseModel):
    evaluations: list[ModelEvaluationOutput]


class CodegenRelevantWarningsEvaluationRequest(BaseModel):
    dataset_name: str
    run_name: str
    run_description: str | None = None
    test: bool = False
    run_on_item_id: str | None = None
    random_for_test: bool = False
    n_runs_per_item: int = 1


class CodegenRelevantWarningsEvaluationSummary(BaseModel):
    started: bool
    item_count: int
    task_ids: list[str]
