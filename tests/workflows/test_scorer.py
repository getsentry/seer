import numpy as np
import pandas as pd
import pytest

from seer.workflows.compare.models import MetricWeights
from seer.workflows.compare.scorer import CohortsMetricsScorer
from seer.workflows.exceptions import ScoringError


@pytest.fixture
def scorer():
    return CohortsMetricsScorer()


@pytest.fixture
def sample_dataset():
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

    result = scorer.kl_metric_lambda(baseline, selection)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    # hardcoded values for KL
    assert np.allclose(result.values, [-0.235, 0.458], atol=1e-3)


def test_compute_metrics(scorer, sample_dataset):
    weights = MetricWeights(kl_divergence_weight=0.7, entropy_weight=0.3)

    result = scorer.compute_metrics(sample_dataset, weights)

    assert "RRF_score" in result.columns
    assert result["RRF_score"].is_monotonic_decreasing


def test_compute_kl_score(scorer, sample_dataset):
    result = scorer.compute_kl_score(sample_dataset)

    assert "kl_individual_scores" in result.columns
    assert "kl_score" in result.columns
    assert all(isinstance(s, dict) for s in result["kl_individual_scores"])
    assert all(isinstance(s, float) for s in result["kl_score"])
    # hardcoded values for KL
    assert np.allclose(result["kl_score"].values, [0.2231, 0.0871], atol=1e-4)


def test_compute_entropy_score(scorer, sample_dataset):
    result = scorer.compute_entropy_score(sample_dataset)
    assert "entropy_score" in result.columns
    assert all(isinstance(s, float) for s in result["entropy_score"])
    # hardcoded values for entropy
    assert np.allclose(result["entropy_score"].values, [0.5004, 0.6109], atol=1e-4)


def test_compute_rrf_score(scorer, sample_dataset):
    weights = MetricWeights(kl_divergence_weight=0.7, entropy_weight=0.3)

    # First compute KL and entropy scores
    dataset = scorer.compute_kl_score(sample_dataset)
    dataset = scorer.compute_entropy_score(dataset)

    result = scorer.compute_rrf_score(dataset, weights)

    assert "RRF_score" in result.columns
    assert "KL_rank" not in result.columns  # Should be dropped
    assert "entropy_rank" not in result.columns  # Should be dropped
    assert result["RRF_score"].is_monotonic_decreasing


def test_error_handling(scorer):
    bad_dataset = pd.DataFrame(
        {
            "attribute_name": ["attr1"],
            "distribution_baseline": [{"A": "not_a_number"}],
            "distribution_selection": [{"A": 0.5}],
        }
    )

    with pytest.raises(ScoringError):
        scorer.compute_kl_score(bad_dataset)
