from dataclasses import dataclass

import pandas as pd
from scipy.special import rel_entr
from scipy.stats import entropy

from seer.workflows.common.constants import DEFAULT_K_RRF
from seer.workflows.compare.models import MetricWeights


@dataclass
class CohortsMetricsScorer:
    """
    Scores cohort comparisons using multiple metrics including KL divergence and entropy.

    This class implements a reciprocal rank fusion (RRF) approach to combine multiple
    scoring metrics into a final ranking score.

    Attributes:
        k_rrf (int): Constant used in RRF calculation to mitigate the impact of high rankings
    """

    k_rrf: int = DEFAULT_K_RRF

    @staticmethod
    def kl_metric_lambda(baseline, selection):
        """
        Calculate the Kullback-Leibler divergence between baseline and selection distributions.

        Args:
            baseline: Probability distribution of the baseline cohort
            selection: Probability distribution of the selection cohort

        Returns:
            Series containing the KL divergence values for each attribute value

        Notes:
            Uses relative entropy to measure how selection distribution differs from baseline
        """
        return rel_entr(pd.Series(baseline), pd.Series(selection))

    def compute_metrics(self, dataset: pd.DataFrame, metric_weights: MetricWeights) -> pd.DataFrame:
        """
        Compute all metrics for the dataset and combine them using RRF.

        Args:
            dataset: DataFrame containing baseline and selection distributions
            metric_weights: Weights to use when combining KL divergence and entropy scores

        Returns:
            DataFrame with added columns for KL scores, entropy scores, and final RRF score,
            sorted by RRF score in descending order

        Process:
            1. Computes KL divergence scores
            2. Computes entropy scores
            3. Combines scores using RRF with provided weights
        """
        dataset = (
            dataset.pipe(self.compute_kl_score)
            .pipe(self.compute_entropy_score)
            .pipe(self.compute_rrf_score, metric_weights)
        )
        return dataset

    def compute_kl_score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute KL divergence scores for each attribute. Higher KL divergence scores indicate greater difference between distributions.

        Args:
            dataset: DataFrame containing baseline and selection distributions

        Returns:
            DataFrame with added columns:
                - kl_individual_scores: Dictionary of KL divergence scores per value
                - kl_score: Sum of individual KL divergence scores

        Notes:
            Higher KL scores indicate greater difference between distributions
        """
        # these scores are needed to rank the values within each attribute
        dataset["kl_individual_scores"] = dataset.apply(
            lambda row: self.kl_metric_lambda(
                row["distribution_baseline"], row["distribution_selection"]
            ).to_dict(),
            axis=1,
        )
        # sum the scores to get the total KL divergence score for each attribute
        dataset["kl_score"] = dataset["kl_individual_scores"].apply(lambda x: sum(x.values()))
        return dataset

    def compute_entropy_score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entropy scores for the selection distribution. We prefer lower entropy scores, as they indicate more concentrated (less uniform) distributions.

        Args:
            dataset: DataFrame containing selection distributions

        Returns:
            DataFrame with added column:
                - entropy_score: Entropy value for each selection distribution

        Notes:
            Lower entropy indicates more concentrated (less uniform) distributions
        """
        dataset["entropy_score"] = dataset["distribution_selection"].apply(
            lambda x: entropy(pd.Series(x))
        )
        return dataset

    def compute_rrf_score(
        self, dataset: pd.DataFrame, metric_weights: MetricWeights
    ) -> pd.DataFrame:
        """
        Compute the final RRF score combining KL divergence and entropy rankings.

        Args:
            dataset: DataFrame containing KL and entropy scores
            metric_weights: Weights for combining KL divergence and entropy rankings

        Returns:
            DataFrame with added columns:
                - KL_rank: RRF-transformed KL divergence rank
                - entropy_rank: RRF-transformed entropy rank
                - RRF_score: Weighted combination of transformed ranks

        Notes:
            - Higher RRF scores indicate more interesting/significant differences
            - KL ranks are ordered descending (higher is more interesting)
            - Entropy ranks are ordered ascending (lower is more interesting)
            - Intermediate rank columns are dropped from final output
        """
        dataset["KL_rank"] = 1 / (
            self.k_rrf + dataset["kl_score"].rank(method="min", ascending=False)
        )
        dataset["entropy_rank"] = 1 / (
            self.k_rrf + dataset["entropy_score"].rank(method="min", ascending=True)
        )
        dataset["RRF_score"] = (
            metric_weights.kl_divergence_weight * dataset["KL_rank"]
            + metric_weights.entropy_weight * dataset["entropy_rank"]
        )
        # drop intermediate rank columns as they are no longer needed
        dataset.drop(columns=["KL_rank", "entropy_rank"], inplace=True)
        # sort the dataset by RRF score in descending order
        return dataset.sort_values(by="RRF_score", ascending=False)
