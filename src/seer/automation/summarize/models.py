from pydantic import BaseModel

from seer.automation.models import IssueDetails


class SummarizeIssueRequest(BaseModel):
    group_id: int
    issue: IssueDetails


class SummarizeIssueResponse(BaseModel):
    group_id: int
    headline: str
    summary: str
    impact: str
