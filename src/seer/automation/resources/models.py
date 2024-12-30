from pydantic import BaseModel

from seer.automation.models import IssueDetails


class Citation(BaseModel):
    url: str
    title: str


class Resource(BaseModel):
    text: str
    citations: list[Citation]


class FindIssueResourcesRequest(BaseModel):
    group_id: int
    issue: IssueDetails
    organization_id: int | None = None
    organization_slug: str | None = None
    project_id: int | None = None


class FindIssueResourcesResponse(BaseModel):
    group_id: int
    resources: list[Resource]
