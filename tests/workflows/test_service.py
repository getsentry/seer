import json
from unittest.mock import Mock

import pytest

from seer.workflows.compare.models import (
    AttributeDistributions,
    CompareCohortsConfig,
    CompareCohortsRequest,
    MetricWeights,
    StatsAttribute,
    StatsAttributeBucket,
    StatsCohort,
)
from seer.workflows.compare.service import CompareService, compareCohorts


@pytest.fixture
def mockProcessor():
    return Mock()


@pytest.fixture
def mockScorer():
    return Mock()


@pytest.fixture
def service(mockProcessor, mockScorer):
    return CompareService(mockProcessor, mockScorer)


@pytest.fixture
def sample_request():
    baseline = StatsCohort(
        totalCount=100,
        attributeDistributions=AttributeDistributions(
            attributes=[
                StatsAttribute(
                    attributeName="attr1",
                    buckets=[
                        StatsAttributeBucket(label="A", value=50),
                        StatsAttributeBucket(label="B", value=50),
                    ],
                )
            ],
        ),
    )

    selection = StatsCohort(
        totalCount=100,
        attributeDistributions=AttributeDistributions(
            attributes=[
                StatsAttribute(
                    attributeName="attr1",
                    buckets=[
                        StatsAttributeBucket(label="A", value=80),
                        StatsAttributeBucket(label="B", value=20),
                    ],
                )
            ],
        ),
    )

    return CompareCohortsRequest(
        baseline=baseline,
        selection=selection,
        config=CompareCohortsConfig(
            metricWeights=MetricWeights(klDivergenceWeight=0.7, entropyWeight=0.3),
            topKAttributes=5,
            topKBuckets=3,
        ),
    )


def test_sanity_check():
    """
    Sanity check to ensure that the results make sense. The test payload contains the syntethic example from the tech spec in the Notion doc.
    """
    with open("tests/workflows/test_data/test_payload_0.json", "r") as f:
        sampleRequest = CompareCohortsRequest(**json.load(f))
    response = compareCohorts(sampleRequest)
    resultAttributes = [attr.attributeName for attr in response.results]
    # browser is the most distrub, followed by country
    expectedAttributes = ["browser", "country"]

    assert resultAttributes == expectedAttributes
