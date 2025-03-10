from dataclasses import dataclass

import pandas as pd

from seer.workflows.models import StatsCohort


@dataclass
class DataProcessor:
    EMPTY_VALUE_ATTRIBUTE = "EMTPY_VALUE"
    alpha = 10**-6

    def preprocess_data(self, data: StatsCohort) -> pd.DataFrame:
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

        # Apply normalization to each Series in the attribute_distribution column
        df["distribution"] = df["distribution"].apply(lambda x: self.add_unseen_value(x))
        return df

    def add_unseen_value(self, distribution: dict) -> pd.Series:
        total_sum = sum(distribution.values())
        if total_sum < 1:
            distribution[self.EMPTY_VALUE_ATTRIBUTE] = 1 - total_sum
        return distribution

    def transform_distribution(self, distribution: pd.Series, all_keys: list) -> dict:
        # reindex distribution to include all keys, filling missing values with 0
        distribution = distribution.reindex(all_keys, fill_value=0)

        # perform lapalce smoothing of the distribution
        # Add alpha to all values and renormalize
        distribution = distribution + self.alpha
        return dict(distribution / distribution.sum())

    def prepare_data(self, data: StatsCohort) -> pd.DataFrame:
        baseline = self.preprocess_data(data.baseline)
        selection = self.preprocess_data(data.selection)

        dataset = baseline.merge(
            selection, on="attribute_name", how="inner", suffixes=("_baseline", "_selection")
        )
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
        dataset.drop(columns=["common_keys"], inplace=True)
        return dataset
