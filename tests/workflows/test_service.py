from unittest.mock import Mock

import pytest

from seer.workflows.compare.models import (
    AttributeDistributions,
    CompareCohortsRequest,
    CompareCohortsResponse,
    MetricWeights,
    Options,
    StatsAttribute,
    StatsAttributeBucket,
    StatsCohort,
)
from seer.workflows.compare.service import CompareService, compare_cohorts


@pytest.fixture
def mock_processor():
    return Mock()


@pytest.fixture
def mock_scorer():
    return Mock()


@pytest.fixture
def service(mock_processor, mock_scorer):
    instance = CompareService.get_instance()
    instance.processor = mock_processor
    instance.scorer = mock_scorer
    return instance


@pytest.fixture
def sample_request():
    baseline = StatsCohort(
        total_count=100,
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
        total_count=100,
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
        options=Options(
            metric_weights=MetricWeights(kl_divergence_weight=0.7, entropy_weight=0.3),
            top_k_attributes=5,
            top_k_buckets=3,
        ),
    )


def test_singleton_pattern():
    # First instance
    service1 = CompareService.get_instance()

    # Second instance should be the same object
    service2 = CompareService.get_instance()

    assert service1 is service2

    # Direct instantiation should raise error
    with pytest.raises(RuntimeError):
        CompareService()


def test_convenience_function(sample_request):
    result = compare_cohorts(sample_request)

    assert isinstance(result, CompareCohortsResponse)
