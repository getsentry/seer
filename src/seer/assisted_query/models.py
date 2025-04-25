from pydantic import BaseModel


class CreateCacheRequest(BaseModel):
    organization_slug: str
    project_ids: list[int]


class CreateCacheResponse(BaseModel):
    success: bool
    message: str
    cache_name: str
