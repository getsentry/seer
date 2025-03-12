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
def sample_cohort():
    return StatsCohort(
        total_count=100,
        attributeDistributions=AttributeDistributions(
            attributes=[
                StatsAttribute(
                    attributeName="test_attr",
                    buckets=[
                        StatsAttributeBucket(label="A", value=50),
                        StatsAttributeBucket(label="B", value=30),
                    ],
                )
            ],
        ),
    )


def test_preprocess_cohort_success(processor, sample_cohort):
    result = processor.preprocess_cohort(sample_cohort)

    expected = pd.DataFrame(
        [
            {
                "attribute_name": "test_attr",
                "distribution": {"A": 0.5, "B": 0.3, EMPTY_VALUE_ATTRIBUTE: 0.2},
            }
        ]
    )

    assert_frame_equal(result, expected)


def test_preprocess_cohort_error(processor):
    with pytest.raises(DataProcessingError):
        processor.preprocess_cohort(None)


def test_add_unseen_value(processor):
    distribution = {"A": 0.5, "B": 0.3}
    result = processor.add_unseen_value(distribution)

    assert result[EMPTY_VALUE_ATTRIBUTE] == pytest.approx(0.2)
    assert sum(result.values()) == pytest.approx(1.0)


def test_transform_distribution(processor):
    distribution = pd.Series({"A": 0.5, "B": 0.3})
    all_keys = ["A", "B", "C"]

    result = processor.transform_distribution(distribution, all_keys)

    # Check that all keys are present
    assert set(result.keys()) == set(all_keys)

    # Check that values sum to 1
    assert sum(result.values()) == pytest.approx(1.0)

    # Check that Laplace smoothing was applied
    assert all(v > 0 for v in result.values())


def test_prepare_data(processor):
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
                        StatsAttributeBucket(label="B", value=80),
                        StatsAttributeBucket(label="C", value=20),
                    ],
                )
            ],
        ),
    )

    request = CompareCohortsRequest(baseline=baseline, selection=selection)
    result = processor.prepare_cohorts_data(request)

    # Check the structure of the result
    assert "attribute_name" in result.columns
    assert "distribution_baseline" in result.columns
    assert "distribution_selection" in result.columns

    # Check that distributions are properly normalized
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distribution_baseline"])
    assert all(sum(d.values()) == pytest.approx(1.0) for d in result["distribution_selection"])
