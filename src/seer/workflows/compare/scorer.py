from dataclasses import dataclass

import pandas as pd
from scipy.special import rel_entr
from scipy.stats import entropy

from seer.workflows.common.constants import DEFAULT_K_RRF
from seer.workflows.compare.models import MetricWeights


@dataclass
class CohortsMetricsScorer:
    k_rrf: int = DEFAULT_K_RRF

    @staticmethod
    def kl_metric_lambda(baseline, selection):
        return rel_entr(pd.Series(baseline), pd.Series(selection))

    def compute_metrics(self, dataset: pd.DataFrame, metric_weights: MetricWeights) -> pd.DataFrame:
        dataset = (
            dataset.pipe(self.compute_kl_score)
            .pipe(self.compute_entropy_score)
            .pipe(self.compute_rrf_score, metric_weights)
        )
        return dataset

    def compute_kl_score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["kl_individual_scores"] = dataset.apply(
            lambda row: self.kl_metric_lambda(
                row["distribution_baseline"], row["distribution_selection"]
            ).to_dict(),
            axis=1,
        )
        dataset["kl_score"] = dataset["kl_individual_scores"].apply(lambda x: sum(x.values()))
        return dataset

    def compute_entropy_score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["entropy_score"] = dataset["distribution_selection"].apply(
            lambda x: entropy(pd.Series(x))
        )
        return dataset

    def compute_rrf_score(
        self, dataset: pd.DataFrame, metric_weights: MetricWeights
    ) -> pd.DataFrame:
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
        dataset.drop(columns=["KL_rank", "entropy_rank"], inplace=True)
        return dataset.sort_values(by="RRF_score", ascending=False)
