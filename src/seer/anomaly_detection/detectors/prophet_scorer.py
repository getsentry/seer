import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ProphetScorer(BaseModel):
    def batch_score(self, df: pd.DataFrame) -> pd.DataFrame:
        return NotImplemented


class ProphetScaledSmoothedScorer(ProphetScorer):
    """
    Generate anomaly scores by comparing observed values to
    yhat_lower and yhat_upper (confidence interval) and
    scaling by standard deviation.
    """

    low_threshold: float = Field(default=0.0, description="The low threshold for the anomaly score")

    high_threshold: float = Field(
        default=0.3, description="The high threshold for the anomaly score"
    )

    def batch_score(self, df: pd.DataFrame):
        """
        Identify anomalies by smoothing scores (10 period moving avg) and
        then comparing results to preset thresholds.

        Args:
            df: dataframe containing forecast and confidence intervals

        Returns:
            Dataframe with anomaly scores data added to it
        """

        # score is the delta between the closest bound and the y-value
        df["score"] = (
            np.where(df["y"] >= df["yhat"], df["y"] - df["yhat_upper"], df["yhat_lower"] - df["y"])
            / df["y"].std()
        )

        # final score is the 10 day rolling average of score
        df["final_score"] = df["score"].rolling(10, center=True, min_periods=1).mean()

        # anomalies: 1 - low confidence, 2 - high confidence, None - normal
        df["anomalies"] = np.where(
            (df["final_score"] >= self.high_threshold) & (df["score"] > 0.0),  # type: ignore
            2.0,
            np.where((df["final_score"] >= self.low_threshold) & (df["score"] > 0.0), 1.0, None),  # type: ignore
        )

        df["flag"] = np.where(df["anomalies"] == 2.0, "anomaly_higher_confidence", "none")

        return df
