from typing import List, Optional

from pydantic import BaseModel, Field

from seer.workflows.common.constants import DEFAULT_ENTROPY_WEIGHT, DEFAULT_KL_DIVERGENCE_WEIGHT


class MetricWeights(BaseModel):
    klDivergenceWeight: float = DEFAULT_KL_DIVERGENCE_WEIGHT
    entropyWeight: float = DEFAULT_ENTROPY_WEIGHT


# Input models
class StatsAttributeBucket(BaseModel):
    # the value of the attribute like "chrome"
    attributeValue: str
    # the count of this attribute value in the cohort like 100.0
    attributeValueCount: float


class StatsAttribute(BaseModel):
    # the name of the attribute like "browser"
    attributeName: str
    # the buckets of the attribute like [StatsAttributeBucket(attributeValue="chrome", attributeValueCount=100.0), StatsAttributeBucket(attributeValue="firefox", attributeValueCount=50.0)]
    buckets: List[StatsAttributeBucket]


class AttributeDistributions(BaseModel):
    attributes: List[StatsAttribute]


class StatsCohort(BaseModel):
    totalCount: float
    attributeDistributions: AttributeDistributions


class Options(BaseModel):
    metricWeights: Optional[MetricWeights] = Field(default_factory=MetricWeights)
    topKAttributes: int | None = None
    topKBuckets: int | None = None


# Request model
class CompareCohortsRequest(BaseModel):
    baseline: StatsCohort
    selection: StatsCohort
    options: Options = Field(default_factory=Options)


# Output models
class AttributeResult(BaseModel):
    # the name of the attribute like "browser"
    attributeName: str
    # the most suspcious values of the attribute like ["chrome", "firefox", "edge"]
    attributeValues: List[str]
    # the score measuring how suspcious the attribute is
    attributeScore: float


# Response model
class CompareCohortsResponse(BaseModel):
    # the list of attributes and their most suspcious values
    results: List[AttributeResult]
