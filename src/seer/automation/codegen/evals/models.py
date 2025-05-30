from pydantic import BaseModel, field_validator
from pydantic.fields import Field

from seer.automation.codebase.models import StaticAnalysisWarning
from seer.automation.codegen.models import CodegenRelevantWarningsRequest
from seer.automation.models import IssueDetails, RepoDefinition


class RepoInfo(BaseModel):
    provider: str
    owner: str
    name: str
    external_id: str

    def to_repo_definition(self, base_commit_sha: str | None = None) -> RepoDefinition:
        return RepoDefinition(
            provider=self.provider,
            owner=self.owner,
            name=self.name,
            external_id=self.external_id,
            base_commit_sha=base_commit_sha,
        )


class EvalItemInput(BaseModel):
    """
    An item in the evaluation dataset.
    """

    repo: RepoDefinition | RepoInfo
    pr_id: int
    more_readable_repos: list[RepoDefinition] | list[RepoInfo] = Field(default_factory=list)
    organization_id: int
    commit_sha: str
    warnings: list[StaticAnalysisWarning]
    issues: list[IssueDetails] | None = None

    @field_validator("issues", mode="after")
    def set_issues_to_empty_list_if_none(cls, v):
        if v is None:
            return []
        return v

    def get_request(self) -> CodegenRelevantWarningsRequest:
        if isinstance(self.repo, RepoDefinition):
            repo_definition = self.repo
        else:
            repo_definition = self.repo.to_repo_definition(base_commit_sha=self.commit_sha)
        more_readable_repos = [
            repo.to_repo_definition() if isinstance(repo, RepoInfo) else repo
            for repo in self.more_readable_repos
        ]
        return CodegenRelevantWarningsRequest(
            repo=repo_definition,
            pr_id=self.pr_id,
            more_readable_repos=more_readable_repos,
            organization_id=self.organization_id,
            warnings=self.warnings,
            callback_url="",
        )


class EvalItemOutput(BaseModel):
    repos: list[str]
    description: str
    encoded_location: str


class ModelEvaluationOutput(BaseModel):
    suggestion_idx: int = Field(description="The index of the suggestion that matches the bug")
    bug_matched_idx: int = Field(
        description="The index of the known bug that was matched to the suggestion"
    )
    match_score: float = Field(
        description="The score for the match between the suggestion and the actual bug, from 0 to 1 (e.g. 0.512)"
    )
    reasoning: str = Field(description="A short explanation of the match score")
    location_match: float = Field(
        description="The score for the location match between the suggestion and the actual bug, from 0 to 1 (e.g. 0.512)"
    )


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
