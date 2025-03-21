import numpy as np
import pandas as pd
import pytest

from seer.workflows.compare.models import CompareCohortsConfig
from seer.workflows.compare.scorer import CohortsMetricsScorer


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
            "attribute_name": ["attr1", "attr2"],
            "distribution_baseline": [{"A": 0.5, "B": 0.5}, {"X": 0.5, "Y": 0.5}],
            "distribution_selection": [{"A": 0.8, "B": 0.2}, {"X": 0.7, "Y": 0.3}],
        }
    )


def test_kl_metric_lambda(scorer):
    baseline = pd.Series({"A": 0.5, "B": 0.5})
    selection = pd.Series({"A": 0.8, "B": 0.2})

    result = scorer._kl_metric_lambda(baseline, selection)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    # hardcoded values for KL
    assert np.allclose(result.values, [-0.235, 0.458], atol=1e-3)


def test_compute_metrics(scorer, sampleDataset, config):
    result = scorer.compute_metrics(sampleDataset, config)

    assert "rrf_score" in result.columns
    assert result["rrf_score"].is_monotonic_decreasing


def test_compute_kl_score(scorer, sampleDataset):
    result = scorer._compute_kl_score(sampleDataset)

    assert "kl_individual_scores" in result.columns
    assert "kl_score" in result.columns
    assert all(isinstance(s, dict) for s in result["kl_individual_scores"])
    assert all(isinstance(s, float) for s in result["kl_score"])
    # hardcoded values for KL
    assert np.allclose(result["kl_score"].values, [0.2231, 0.0871], atol=1e-4)


def test_compute_entropy_score(scorer, sampleDataset):
    result = scorer._compute_entropy_score(sampleDataset)
    assert "entropy_score" in result.columns
    assert all(isinstance(s, float) for s in result["entropy_score"])
    # hardcoded values for entropy
    assert np.allclose(result["entropy_score"].values, [0.5004, 0.6109], atol=1e-4)


def test_compute_rrf_score(scorer, sampleDataset, config):

    # First compute KL and entropy scores
    dataset = scorer._compute_kl_score(sampleDataset)
    dataset = scorer._compute_entropy_score(dataset)

    result = scorer._compute_rrf_score(dataset, config)

    assert "rrf_score" in result.columns
    assert "kl_rank" not in result.columns  # Should be dropped
    assert "entropy_rank" not in result.columns  # Should be dropped
    assert result["rrf_score"].is_monotonic_decreasing
