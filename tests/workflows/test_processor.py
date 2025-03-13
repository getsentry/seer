import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from seer.workflows.compare.models import (
    AttributeDistributions,
    CompareCohortsConfig,
    CompareCohortsRequest,
    StatsAttribute,
    StatsAttributeBucket,
    StatsCohort,
)
from seer.workflows.compare.processor import DataProcessor
from seer.workflows.exceptions import DataProcessingError


@pytest.fixture
def processor():
    return DataProcessor()


@pytest.fixture
def config():
    return CompareCohortsConfig()


@pytest.fixture
def sampleCohort():
    return StatsCohort(
        totalCount=100,
        attributeDistributions=AttributeDistributions(
            attributes=[
                StatsAttribute(
                    attributeName="test_attr",
                    buckets=[
                        StatsAttributeBucket(attributeValue="A", attributeValueCount=50),
                        StatsAttributeBucket(attributeValue="B", attributeValueCount=30),
                    ],
                )
            ],
        ),
    )


def test_preprocessCohortSuccess(processor, sampleCohort, config):
    result = processor._preprocessCohort(sampleCohort, config)

    expected = pd.DataFrame(
        [
            {
                "attributeName": "test_attr",
                "distribution": {"A": 0.5, "B": 0.3, config.emptyValueAttribute: 0.2},
            }
        ]
    )

    assert_frame_equal(result, expected)


def test_preprocessCohortError(processor, config):
    with pytest.raises(DataProcessingError):
        processor._preprocessCohort(None, config)


def test_addUnseenValue(processor, config):
    distribution = {"A": 0.5, "B": 0.3}
    result = processor._addUnseenValue(distribution, config)

    assert result[config.emptyValueAttribute] == pytest.approx(0.2)
    assert sum(result.values()) == pytest.approx(1.0)


def test_transformDistribution(processor, config):
    distribution = pd.Series({"A": 0.5, "B": 0.3})
    allKeys = ["A", "B", "C"]

    result = processor._transformDistribution(distribution, allKeys, config)

    # Check that all keys are present
    assert set(result.keys()) == set(allKeys)

    # Check that values sum to 1
    assert sum(result.values()) == pytest.approx(1.0)

    # Check that Laplace smoothing was applied
    assert all(v > 0 for v in result.values())


def test_prepareCohortsData(processor, config):
    baseline = StatsCohort(
        totalCount=100,
        attributeDistributions=AttributeDistributions(
            attributes=[
                StatsAttribute(
                    attributeName="attr1",
                    buckets=[
                        StatsAttributeBucket(attributeValue="A", attributeValueCount=50),
                        StatsAttributeBucket(attributeValue="B", attributeValueCount=50),
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
                        StatsAttributeBucket(attributeValue="B", attributeValueCount=80),
                        StatsAttributeBucket(attributeValue="C", attributeValueCount=20),
                    ],
                )
            ],
        ),
    )

    request = CompareCohortsRequest(baseline=baseline, selection=selection, config=config)
    result = processor.prepareCohortsData(request)

    # Check the structure of the result
    assert "attributeName" in result.columns
    assert "distributionBaseline" in result.columns
    assert "distributionSelection" in result.columns

    # Check that distributions are properly normalized
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distributionBaseline"])
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distributionSelection"])
