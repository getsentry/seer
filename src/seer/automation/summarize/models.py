from typing import Optional

from pydantic import BaseModel

from seer.automation.models import IssueDetails


class SummarizeIssueRequest(BaseModel):
    group_id: int
    issue: IssueDetails
    organization_id: int | None = None
    organization_slug: str | None = None
    project_id: int | None = None
    connected_issues: Optional[list[IssueDetails]] = None


class SummarizeIssueScores(BaseModel):
    possible_cause_confidence: float
    possible_cause_novelty: float


class SummarizeIssueResponse(BaseModel):
    group_id: int
    headline: str
    whats_wrong: str
    trace: str
    possible_cause: str
    scores: SummarizeIssueScores | None = None
