from typing import List, Optional

from pydantic import BaseModel, Field


# Input models
class StatsAttributeBucket(BaseModel):
    label: str
    value: float


class StatsAttribute(BaseModel):
    attributeName: str
    buckets: List[StatsAttributeBucket]


class AttributeDistributions(BaseModel):
    total_count: float
    attributes: List[StatsAttribute]


class StatsCohort(BaseModel):
    attributeDistributions: AttributeDistributions


class MetricWeights(BaseModel):
    kl_divergence_weight: float = 0.8
    entropy_weight: float = 0.2


class Options(BaseModel):
    metric_weights: Optional[MetricWeights] = Field(default_factory=MetricWeights)
    top_k_attributes: int | None = None
    top_k_buckets: int | None = None


# Request model
class CompareCohortsRequest(BaseModel):
    baseline: StatsCohort
    selection: StatsCohort
    options: Options | None = None


# Output models
class AttributeResult(BaseModel):
    attributeName: str
    attributeValues: List[str]


# Response model
class CompareCohortsResponse(BaseModel):
    results: List[AttributeResult]
