from pydantic import BaseModel


class CreateCacheRequest(BaseModel):
    org_id: int
    project_ids: list[int]
    no_values: bool = False


class CreateCacheResponse(BaseModel):
    success: bool
    message: str
    cache_name: str


class Chart(BaseModel):
    chart_type: int
    y_axes: list[str]


class TranslateRequest(BaseModel):
    org_id: int
    project_ids: list[int]
    natural_language_query: str


class TranslateResponse(BaseModel):
    query: str
    stats_period: str
    group_by: list[str]
    visualization: list[Chart]
    sort: str


class TranslateResponses(BaseModel):
    responses: list[TranslateResponse]


class ModelResponse(BaseModel):
    explanation: str
    query: str
    stats_period: str
    group_by: list[str]
    visualization: list[Chart]
    sort: str
    confidence_score: float


class RelevantFieldsResponse(BaseModel):
    fields: list[str]


class ValuesResponse(BaseModel):
    values: list[str]
