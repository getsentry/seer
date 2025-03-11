from dataclasses import dataclass

import pandas as pd

from seer.workflows.common.constants import DEFAULT_ALPHA, EMPTY_VALUE_ATTRIBUTE
from seer.workflows.compare.models import CompareCohortsRequest, StatsCohort
from seer.workflows.exceptions import DataProcessingError


@dataclass
class DataProcessor:
    """
    Processes cohort data by normalizing and transforming attribute distributions.

    Attributes:
        empty_value_attribute (str): Label used for missing values in distributions
        alpha (float): Smoothing parameter for Laplace smoothing
    """

    empty_value_attribute: str = EMPTY_VALUE_ATTRIBUTE
    alpha: float = DEFAULT_ALPHA

    def preprocess_cohort(self, data: StatsCohort) -> pd.DataFrame:
        """
        Preprocess a single cohort's attribute distributions into a normalized DataFrame with added unseen value

        Args:
            data: StatsCohort object containing attribute distributions

        Returns:
            pd.DataFrame: DataFrame with columns:
                - attribute_name: Name of the attribute
                - distribution: Dictionary mapping labels to normalized values
        """
        try:
            df = pd.DataFrame(
                [
                    {
                        "attribute_name": attr.attributeName,
                        "distribution": {
                            item.label: item.value / data.attributeDistributions.total_count
                            for item in attr.buckets
                        },
                    }
                    for attr in data.attributeDistributions.attributes
                ]
            )
            df["distribution"] = df["distribution"].apply(lambda x: self.add_unseen_value(x))
            return df
        except Exception as e:
            raise DataProcessingError(f"Failed to preprocess cohort data: {str(e)}") from e

    def add_unseen_value(self, distribution: dict[str, float]) -> dict[str, float]:
        """
        Add an unseen value to the distribution if the total probability is less than 1. This can happen when the total count of the attribute is less than the total count of the cohort.

        Args:
            distribution: Dictionary mapping labels to probability values

        Returns:
            Updated distribution with unseen value added if needed
        """
        total_sum = sum(distribution.values())
        # if the total probability is less than 1, add an unseen value to the distribution
        if total_sum < 1:
            distribution[self.empty_value_attribute] = 1 - total_sum
        return distribution

    def transform_distribution(
        self, distribution: pd.Series[float], all_keys: list[str]
    ) -> dict[str, float]:
        """
        Apply Laplace smoothing to a probability distribution. It's needed to avoid zero probabilities, which would break the KL divergence calculation.

        Args:
            distribution: Series containing the probability distribution
            all_keys: List of all possible keys that should be in the distribution (extracted from the two cohorts)

        Returns:
            Dictionary containing the smoothed distribution with all keys present

        Notes:
            - Missing keys are filled with 0 before smoothing
            - Adds alpha to all values and renormalizes to maintain sum = 1
        """
        try:
            # reindex distribution to include all keys, filling missing values with 0
            distribution = distribution.reindex(all_keys, fill_value=0)

            # perform lapalce smoothing of the distribution
            # Add alpha to all values and renormalize
            distribution = distribution + self.alpha
            return dict(distribution / distribution.sum())
        except Exception as e:
            raise DataProcessingError(f"Failed to transform distribution: {str(e)}") from e

    def prepare_cohorts_data(self, request: CompareCohortsRequest) -> pd.DataFrame:
        """
        Prepare and combine baseline and selection cohort data for comparison.

        Args:
            request: CompareCohortsRequest containing both baseline and selection cohorts

        Returns:
            pd.DataFrame: DataFrame with columns:
                - attribute_name: Name of the attribute
                - distribution_baseline: Smoothed baseline distribution
                - distribution_selection: Smoothed selection distribution

        Process:
            1. Preprocesses both baseline and selection data
            2. Merges the datasets on attribute_name
            3. Identifies common keys across distributions
            4. Applies Laplace smoothing to both distributions
            5. Cleans up intermediate calculation columns
        """

        baseline = self.preprocess_cohort(request.baseline)
        selection = self.preprocess_cohort(request.selection)

        dataset = baseline.merge(
            selection, on="attribute_name", how="inner", suffixes=("_baseline", "_selection")
        )
        # identify common keys which appear in both distributions
        dataset["common_keys"] = dataset.apply(
            lambda row: set(row["distribution_baseline"].keys())
            | set(row["distribution_selection"].keys()),
            axis=1,
        )

        for col in ["distribution_baseline", "distribution_selection"]:
            dataset[col] = dataset.apply(
                lambda row: self.transform_distribution(pd.Series(row[col]), row["common_keys"]),
                axis=1,
            )
        # drop the common_keys column as it's no longer needed
        dataset.drop(columns=["common_keys"], inplace=True)
        return dataset
