import pandas as pd

from seer.workflows.compare.models import CompareCohortsConfig, CompareCohortsRequest, StatsCohort
from seer.workflows.exceptions import DataProcessingError


class DataProcessor:
    """
    Processes cohort data by normalizing and transforming attribute distributions.
    """

    def _preprocessCohort(self, data: StatsCohort, config: CompareCohortsConfig) -> pd.DataFrame:
        """
        Preprocess a single cohort's attribute distributions into a normalized DataFrame with added unseen value

        Args:
            data: StatsCohort object containing attribute distributions
            config: CompareCohortsConfig containing the configuration for the comparison
        Returns:
            pd.DataFrame: DataFrame with columns:
                - attributeName: Name of the attribute
                - distribution: Dictionary mapping labels to normalized values
        """
        try:
            df = pd.DataFrame(
                [
                    {
                        "attributeName": attr.attributeName,
                        "distribution": {
                            item.attributeValue: item.attributeValueCount / data.totalCount
                            for item in attr.buckets
                        },
                    }
                    for attr in data.attributeDistributions.attributes
                ]
            )
            df["distribution"] = df["distribution"].apply(lambda x: self._addUnseenValue(x, config))
            return df
        except Exception as e:
            raise DataProcessingError(f"Failed to preprocess cohort data: {e}") from e

    def _addUnseenValue(
        self, distribution: dict[str, float], config: CompareCohortsConfig
    ) -> dict[str, float]:
        """
        Add an unseen value to the distribution if the total probability is less than 1. This can happen when the total count of the attribute is less than the total count of the cohort.

        Args:
            distribution: Dictionary mapping labels to probability values
            config: CompareCohortsConfig containing the configuration for the comparison
        Returns:
            Updated distribution with unseen value added if needed
        """
        totalSum = sum(distribution.values())
        # if the total probability is less than 1, add an unseen value to the distribution
        if totalSum < 1:
            distribution[config.emptyValueAttribute] = 1 - totalSum
        return distribution

    def _transformDistribution(
        self, distribution: pd.Series, allKeys: list[str], config: CompareCohortsConfig
    ) -> dict[str, float]:
        """
        Apply Laplace smoothing to a probability distribution. It's needed to avoid zero probabilities, which would break the KL divergence calculation.

        Args:
            distribution: Series containing the probability distribution
            allKeys: List of all possible keys that should be in the distribution (extracted from the two cohorts)
            config: CompareCohortsConfig containing the configuration for the comparison
        Returns:
            Dictionary containing the smoothed distribution with all keys present

        Notes:
            - Missing keys are filled with 0 before smoothing
            - Adds alpha to all values and renormalizes to maintain sum = 1
        """
        try:
            # reindex distribution to include all keys, filling missing values with 0
            distribution = distribution.reindex(allKeys, fill_value=0)

            # perform lapalce smoothing of the distribution
            # Add alpha to all values and renormalize
            distribution = distribution + config.alphaLaplace
            return dict(distribution / distribution.sum())
        except Exception as e:
            raise DataProcessingError(f"Failed to transform distribution: {str(e)}") from e

    def prepareCohortsData(self, request: CompareCohortsRequest) -> pd.DataFrame:
        """
        Prepare and combine baseline and selection cohort data for comparison.

        Args:
            request: CompareCohortsRequest containing both baseline and selection cohorts
        Returns:
            pd.DataFrame: DataFrame with columns:
                - attributeName: Name of the attribute
                - distributionBaseline: Smoothed baseline distribution
                - distributionSelection: Smoothed selection distribution

        Process:
            1. Preprocesses both baseline and selection data
            2. Merges the datasets on attributeName
            3. Identifies common keys across distributions
            4. Applies Laplace smoothing to both distributions
            5. Cleans up intermediate calculation columns
        """
        config = request.config
        baseline = self._preprocessCohort(request.baseline, config)
        selection = self._preprocessCohort(request.selection, config)

        dataset = baseline.merge(
            selection, on="attributeName", how="inner", suffixes=("Baseline", "Selection")
        )
        # identify common keys which appear in both distributions
        dataset["commonAttributeValues"] = dataset.apply(
            lambda row: set(row["distributionBaseline"].keys())
            | set(row["distributionSelection"].keys()),
            axis=1,
        )

        for col in ["distributionBaseline", "distributionSelection"]:
            dataset[col] = dataset.apply(
                lambda row: self._transformDistribution(
                    pd.Series(row[col]), row["commonAttributeValues"], config
                ),
                axis=1,
            )
        # drop the commonKeys column as it's no longer needed
        dataset.drop(columns=["commonAttributeValues"], inplace=True)
        return dataset
