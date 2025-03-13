import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from seer.workflows.common.constants import EMPTY_VALUE_ATTRIBUTE
from seer.workflows.compare.models import (
    AttributeDistributions,
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


def test_preprocessCohortSuccess(processor, sampleCohort):
    result = processor.preprocessCohort(sampleCohort)

    expected = pd.DataFrame(
        [
            {
                "attributeName": "test_attr",
                "distribution": {"A": 0.5, "B": 0.3, EMPTY_VALUE_ATTRIBUTE: 0.2},
            }
        ]
    )

    assert_frame_equal(result, expected)


def test_preprocessCohortError(processor):
    with pytest.raises(DataProcessingError):
        processor.preprocessCohort(None)


def test_addUnseenValue(processor):
    distribution = {"A": 0.5, "B": 0.3}
    result = processor.addUnseenValue(distribution)

    assert result[EMPTY_VALUE_ATTRIBUTE] == pytest.approx(0.2)
    assert sum(result.values()) == pytest.approx(1.0)


def test_transformDistribution(processor):
    distribution = pd.Series({"A": 0.5, "B": 0.3})
    allKeys = ["A", "B", "C"]

    result = processor.transformDistribution(distribution, allKeys)

    # Check that all keys are present
    assert set(result.keys()) == set(allKeys)

    # Check that values sum to 1
    assert sum(result.values()) == pytest.approx(1.0)

    # Check that Laplace smoothing was applied
    assert all(v > 0 for v in result.values())


def test_prepareCohortsData(processor):
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

    request = CompareCohortsRequest(baseline=baseline, selection=selection)
    result = processor.prepareCohortsData(request)

    # Check the structure of the result
    assert "attributeName" in result.columns
    assert "distributionBaseline" in result.columns
    assert "distributionSelection" in result.columns

    # Check that distributions are properly normalized
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distributionBaseline"])
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distributionSelection"])
