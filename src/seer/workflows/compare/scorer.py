import pandas as pd
from scipy.special import rel_entr
from scipy.stats import entropy

from seer.workflows.compare.models import CompareCohortsConfig
from seer.workflows.exceptions import ScoringError


class CohortsMetricsScorer:
    """
    Scores cohort comparisons using multiple metrics including KL divergence and entropy.

    This class implements a reciprocal rank fusion (RRF) approach to combine multiple
    scoring metrics into a final ranking score.
    """

    @staticmethod
    def _klMetricLambda(baseline: dict[str, float], selection: dict[str, float]) -> pd.Series:
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

    def computeMetrics(self, dataset: pd.DataFrame, config: CompareCohortsConfig) -> pd.DataFrame:
        """
        Compute all metrics for the dataset and combine them using RRF.

        Args:
            dataset: DataFrame containing baseline and selection distributions
            config: Config for the comparison

        Returns:
            DataFrame with added columns for KL scores, entropy scores, and final RRF score,
            sorted by RRF score in descending order

        Process:
            1. Computes KL divergence scores
            2. Computes entropy scores
            3. Combines scores using RRF with provided weights
        """
        dataset = (
            dataset.pipe(self._computeKLScore)
            .pipe(self._computeEntropyScore)
            .pipe(self._computeRRFScore, config)
        )
        return dataset

    def _computeKLScore(self, dataset: pd.DataFrame) -> pd.DataFrame:
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
        try:
            # these scores are needed to rank the values within each attribute
            dataset["klIndividualScores"] = dataset.apply(
                lambda row: self._klMetricLambda(
                    row["distributionBaseline"], row["distributionSelection"]
                ).to_dict(),
                axis=1,
            )
            # sum the scores to get the total KL divergence score for each attribute
            dataset["klScore"] = dataset["klIndividualScores"].apply(lambda x: sum(x.values()))
            return dataset
        except Exception as e:
            raise ScoringError(f"Failed to compute KL divergence scores: {e}") from e

    def _computeEntropyScore(self, dataset: pd.DataFrame) -> pd.DataFrame:
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
        try:
            dataset["entropyScore"] = dataset["distributionSelection"].apply(
                lambda x: entropy(pd.Series(x))
            )
            return dataset
        except Exception as e:
            raise ScoringError(f"Failed to compute entropy scores: {e}") from e

    def _computeRRFScore(self, dataset: pd.DataFrame, config: CompareCohortsConfig) -> pd.DataFrame:
        """
        Compute the final RRF score combining KL divergence and entropy rankings.

        Args:
            dataset: DataFrame containing KL and entropy scores
            config: CompareCohortsConfig containing the configuration for the comparison
        Returns:
            DataFrame with added columns:
                - klRank: RRF-transformed KL divergence rank
                - entropyRank: RRF-transformed entropy rank
                - rrfScore: Weighted combination of transformed ranks

        Notes:
            - Higher RRF scores indicate more interesting/significant differences
            - KL ranks are ordered descending (higher is more interesting)
            - Entropy ranks are ordered ascending (lower is more interesting)
            - Intermediate rank columns are dropped from final output
        """
        try:
            dataset["klRank"] = 1 / (
                config.kRRF + dataset["klScore"].rank(method="min", ascending=False)
            )
            dataset["entropyRank"] = 1 / (
                config.kRRF + dataset["entropyScore"].rank(method="min", ascending=True)
            )
            dataset["rrfScore"] = (
                config.metricWeights.klDivergenceWeight * dataset["klRank"]
                + config.metricWeights.entropyWeight * dataset["entropyRank"]
            )
            # drop intermediate rank columns as they are no longer needed
            dataset.drop(columns=["klRank", "entropyRank"], inplace=True)
            # sort the dataset by RRF score in descending order
            return dataset.sort_values(by="rrfScore", ascending=False).reset_index(drop=True)
        except Exception as e:
            raise ScoringError(f"Failed to compute RRF score: {e}") from e
