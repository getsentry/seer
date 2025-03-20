import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from seer.workflows.compare.models import (
    AttributeDistributions,
    CompareCohortsConfig,
    CompareCohortsMeta,
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
def meta():
    return CompareCohortsMeta(referrer="test_referrer")


@pytest.fixture
def sample_cohort():
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


def test_preprocess_cohort_success(processor, sample_cohort, config):
    result = processor._preprocess_cohort(sample_cohort, config)

    expected = pd.DataFrame(
        [
            {
                "attribute_name": "test_attr",
                "distribution": {"A": 0.5, "B": 0.3, config.emptyValueAttribute: 0.2},
            }
        ]
    )

    assert_frame_equal(result, expected)


def test_preprocess_cohort_error(processor, config):
    with pytest.raises(DataProcessingError):
        processor._preprocess_cohort(None, config)


def test_add_unseen_value(processor, config):
    distribution = {"A": 0.5, "B": 0.3}
    result = processor._add_unseen_value(distribution, config)

    assert result[config.emptyValueAttribute] == pytest.approx(0.2)
    assert sum(result.values()) == pytest.approx(1.0)


def test_transform_distribution(processor, config):
    distribution = pd.Series({"A": 0.5, "B": 0.3})
    all_keys = ["A", "B", "C"]

    result = processor._transform_distribution(
        distribution=distribution, all_keys=all_keys, total_count=1000, config=config
    )

    # Check that all keys are present
    assert set(result.keys()) == set(all_keys)

    # Check that values sum to 1
    assert sum(result.values()) == pytest.approx(1.0)

    # Check that Laplace smoothing was applied
    assert all(v > 0 for v in result.values())


def test_prepare_cohort_data(processor, config, meta):
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

    request = CompareCohortsRequest(
        baseline=baseline, selection=selection, config=config, meta=meta
    )
    result = processor.prepare_cohort_data(request)
    # Check the structure of the result
    assert "attribute_name" in result.columns
    assert "distribution_baseline" in result.columns
    assert "distribution_selection" in result.columns

    # Check that distributions are properly normalized
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distribution_baseline"])
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distribution_selection"])
    # approximate equality due to Laplace smoothing
    expected_baseline = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    expected_selection = pd.Series({"A": 0.0, "B": 0.8, "C": 0.2})
    assert_series_equal(
        pd.Series(result["distribution_baseline"].iloc[0]).sort_index(),
        expected_baseline,
        atol=config.alphaLaplace,
    )
    assert_series_equal(
        pd.Series(result["distribution_selection"].iloc[0]).sort_index(),
        expected_selection,
        atol=config.alphaLaplace,
    )
