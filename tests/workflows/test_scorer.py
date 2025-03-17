import numpy as np
import pandas as pd
import pytest

from seer.workflows.compare.models import CompareCohortsConfig
from seer.workflows.compare.scorer import CohortsMetricsScorer
from seer.workflows.exceptions import ScoringError


@pytest.fixture
def scorer():
    return CohortsMetricsScorer()


@pytest.fixture
def config():
    return CompareCohortsConfig()


@pytest.fixture
def sampleDataset():
    return pd.DataFrame(
        {
            "attributeName": ["attr1", "attr2"],
            "distributionBaseline": [{"A": 0.5, "B": 0.5}, {"X": 0.5, "Y": 0.5}],
            "distributionSelection": [{"A": 0.8, "B": 0.2}, {"X": 0.7, "Y": 0.3}],
        }
    )


def test_klMetricLambda(scorer):
    baseline = pd.Series({"A": 0.5, "B": 0.5})
    selection = pd.Series({"A": 0.8, "B": 0.2})

    result = scorer._klMetricLambda(baseline, selection)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    # hardcoded values for KL
    assert np.allclose(result.values, [-0.235, 0.458], atol=1e-3)


def test_computeMetrics(scorer, sampleDataset, config):
    result = scorer.computeMetrics(sampleDataset, config)

    assert "rrfScore" in result.columns
    assert result["rrfScore"].is_monotonic_decreasing


def test_computeKLScore(scorer, sampleDataset):
    result = scorer._computeKLScore(sampleDataset)

    assert "klIndividualScores" in result.columns
    assert "klScore" in result.columns
    assert all(isinstance(s, dict) for s in result["klIndividualScores"])
    assert all(isinstance(s, float) for s in result["klScore"])
    # hardcoded values for KL
    assert np.allclose(result["klScore"].values, [0.2231, 0.0871], atol=1e-4)


def test_computeEntropyScore(scorer, sampleDataset):
    result = scorer._computeEntropyScore(sampleDataset)
    assert "entropyScore" in result.columns
    assert all(isinstance(s, float) for s in result["entropyScore"])
    # hardcoded values for entropy
    assert np.allclose(result["entropyScore"].values, [0.5004, 0.6109], atol=1e-4)


def test_computeRRFScore(scorer, sampleDataset, config):

    # First compute KL and entropy scores
    dataset = scorer._computeKLScore(sampleDataset)
    dataset = scorer._computeEntropyScore(dataset)

    result = scorer._computeRRFScore(dataset, config)

    assert "rrfScore" in result.columns
    assert "klRank" not in result.columns  # Should be dropped
    assert "entropyRank" not in result.columns  # Should be dropped
    assert result["rrfScore"].is_monotonic_decreasing


def test_errorHandling(scorer):
    badDataset = pd.DataFrame(
        {
            "attributeName": ["attr1"],
            "distributionBaseline": [{"A": "not_a_number"}],
            "distributionSelection": [{"A": 0.5}],
        }
    )

    with pytest.raises(ScoringError):
        scorer._computeKLScore(badDataset)
