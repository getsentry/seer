from enum import Enum
from typing import Literal

from pydantic import BaseModel


class Chart(BaseModel):
    chart_type: Literal[1, 2, 3]
    y_axes: list[list[str]]


class TranslateRequest(BaseModel):
    organization_slug: str
    project_ids: list[int]
    natural_language_query: str


class TranslateResponse(BaseModel):
    query: str
    stats_period: str
    group_by: str
    visualization: Chart
    sort: str


class ModelResponse(BaseModel):
    explanation: str
    query: str
    stats_period: str
    group_by: str
    visualization: Chart
    sort: str
    confidence_score: float


class RelevantFieldsResponse(BaseModel):
    fields: list[str]


class ValuesResponse(BaseModel):
    values: list[str]


class ModelProvider(Enum):
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
