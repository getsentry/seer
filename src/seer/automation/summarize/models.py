from typing import Optional

from pydantic import BaseModel

from seer.automation.models import EAPTrace, IssueDetails


class SummarizeIssueRequest(BaseModel):
    group_id: int
    issue: IssueDetails
    organization_id: int | None = None
    organization_slug: str | None = None
    project_id: int | None = None
    connected_issues: Optional[list[IssueDetails]] = None


class GetFixabilityScoreRequest(BaseModel):
    group_id: int


class SummarizeIssueScores(BaseModel):
    possible_cause_confidence: float | None = None
    possible_cause_novelty: float | None = None
    fixability_score: float | None = None
    fixability_score_version: int | None = None
    is_fixable: bool | None = None


class SummarizeIssueResponse(BaseModel):
    group_id: int
    headline: str
    whats_wrong: str
    trace: str
    possible_cause: str
    scores: SummarizeIssueScores | None = None


class SummarizeTraceRequest(BaseModel):
    trace_id: str
    only_transactions: bool = False
    trace: EAPTrace


class SummarizeTraceResponse(BaseModel):
    trace_id: str
    summary: str
    key_observations: str
    performance_characteristics: str
    suggested_investigations: str
