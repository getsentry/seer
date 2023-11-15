import logging

import pandas as pd
import numpy as np
from prophet import Prophet
from tsmoothie.smoother import SpectralSmoother
from scipy import stats, special
from datetime import datetime, timedelta

from seer.anomaly_detection.prophet_params import ProphetParams


class ProphetDetector(Prophet):
    """
    Anomaly Detection class built around Prophet.

    The class uses a trained prophet model to build confidence intervals
    around a timeseries, and those confidence intervals to establish
    anomaly scores by comparing the intervals to observed values.

    Attributes:
        params: the parameter class defined with `ProphetParams`

    """

    def __init__(self, params: ProphetParams) -> None:
        self.low_threshold, self.high_threshold = 0.0, 0.2
        self.model_params = params
        super().__init__()

    def pre_process_data(self, data: pd.DataFrame, granularity: int, start: str, end: str):
        """
        Apply kalman filter and log transform input data

        Attributes:
        data: the input time series data (pd.DataFrame)
        start: start time for the anomaly detection window
        end: end time for the anomaly detection window

        Initializes:
            data: training dataset
            test: test dataset
            bc_lambda: box-cox lambda used to undo log transform

        """
        train = data.rename(columns={"time": "ds", "count": "y"})
        train["ds"] = pd.to_datetime(train["ds"]).dt.tz_localize(None)

        # no need to preprocess data if it is constant
        if train["y"].nunique() != 1:
            smoother = SpectralSmoother(smooth_fraction=0.35, pad_len=10)
            smoother.smooth(list(train["y"]))
            train["y"] = smoother.smooth_data[0]
            train["y"] = np.where(smoother.smooth_data[0] < 0, 0, smoother.smooth_data[0])
            train["y"] = self._boxcox(train["y"])

        # we are using zerofill=True, so we need to fill in records even if there is no data
        train = train.set_index("ds", drop=False).asfreq(timedelta(seconds=granularity))
        train["ds"] = train.index

        self.start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        self.end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

        buffer = timedelta(seconds=granularity * 10)
        self.test = train[self.start - buffer : self.end + buffer]
        self.train = train

    def fit(self, **kwargs) -> None:
        """
        Fit Prophet model

        Args:
            None

        Returns:
            The fitted prophet model object
        """
        assert hasattr(
            self, "train"
        ), "Must build training dataset with pre_process_data before model training"
        df = self.train

        prophet = Prophet(
            growth=self.model_params.growth,
            changepoints=self.model_params.changepoints,
            n_changepoints=self.model_params.n_changepoints,
            changepoint_range=self.model_params.changepoint_range,
            yearly_seasonality=self.model_params.yearly_seasonality,
            weekly_seasonality=self.model_params.weekly_seasonality,
            daily_seasonality=self.model_params.daily_seasonality,
            holidays=self.model_params.holidays,
            seasonality_mode=self.model_params.seasonality_mode,
            seasonality_prior_scale=self.model_params.seasonality_prior_scale,
            holidays_prior_scale=self.model_params.holidays_prior_scale,
            changepoint_prior_scale=self.model_params.changepoint_prior_scale,
            mcmc_samples=self.model_params.mcmc_samples,
            interval_width=self.model_params.interval_width,
            uncertainty_samples=self.model_params.uncertainty_samples,
        )

        if self.model_params.growth == "logistic":
            # assign cap to a new col as Prophet required
            df["cap"] = self.model_params.cap

        # Adding floor if available
        if self.model_params.floor is not None:
            df["floor"] = self.model_params.floor

        # Add any specified custom seasonalities.
        for custom_seasonality in self.model_params.custom_seasonalities:
            prophet.add_seasonality(**custom_seasonality)

        # limit iter to 250 to avoid long inference times
        self.model = prophet.fit(df=df, iter=250)
        logging.info("Fitted Prophet model.")

    def predict(self):
        """
        Generate predictions using prophet model

        Args:
            None

        Returns:
            Test dataset with predictions added
        """
        forecast = self.model.predict(self.test)
        forecast["y"] = self.test["y"].fillna(0).values
        forecast.index = forecast["ds"]

        return forecast

    def scale_scores(self, df: pd.DataFrame):
        """
        Generate anomaly scores by comparing observed values to
        yhat_lower and yhat_upper (confidence interval) and
        scaling by standard deviation.

        Identify anomalies by smoothing scores (10 period moving avg) and
        then comparing results to preset thresholds.

        Args:
            df: dataframe containing forecast and confidence intervals

        Returns:
            Dataframe with anomaly scores data added to it
        """

        # score is the delta between the closest bound and the y-value
        df["score"] = (
            np.where(
                df["y"] >= df["yhat"], df["y"] - df["yhat_upper"], df["yhat_lower"] - df["y"]
            )
            / df["y"].std()
        )

        # final score is the 10 day rolling average of score
        df["final_score"] = df["score"].rolling(10, center=True, min_periods=1).mean()

        # anomalies: 1 - low confidence, 2 - high confidence, None - normal
        df["anomalies"] = np.where(
            (df["final_score"] >= self.high_threshold) & (df["score"] > 0),
            2,
            np.where((df["final_score"] >= self.low_threshold) & (df["score"] > 0), 1, None),
        )

        return df[self.start : self.end]

    def add_prophet_uncertainty(self, df: pd.DataFrame):
        """
        Adds yhat_upper and yhat_lower to the forecast_df,
        based on the params of a trained prophet model and the interval_width.

        Args:
            df: DataFrame with predicted values

        Returns:
            DataFrame with confidence intervals (yhat_upper and yhat_lower) added
        """
        assert "yhat" in df.columns, "Must have the mean yhat forecast to build uncertainty on"
        interval_width = self.model_params.interval_width

        # there is no trend-based uncertainty if we're only looking on the past where trend is known
        sample_trends = np.zeros((2000, len(df)))

        # add gaussian noise based on historical levels
        sigma = self.model.params["sigma_obs"][0]
        historical_variance = np.random.normal(scale=sigma, size=sample_trends.shape)
        full_samples = sample_trends + historical_variance
        # get quantiles and scale back (prophet scales the data before fitting, so sigma and deltas are scaled)
        width_split = (1 - interval_width) / 2
        quantiles = np.array([width_split, 1 - width_split]) * 100  # get quantiles from width
        quantiles = np.percentile(full_samples, quantiles, axis=0)
        # Prophet scales all the data before fitting and predicting, y_scale re-scales it to original values
        quantiles = quantiles * self.model.y_scale

        df["yhat_lower"] = quantiles[0] + df.yhat
        df["yhat_upper"] = quantiles[1] + df.yhat

        should_invert = True if df["y"].nunique() != 1 else False
        for col in ["y", "yhat", "yhat_lower", "yhat_upper"]:
            df[col] = np.where(df[col] < 0.0, 0.0, df[col])
            if should_invert:
                df[col] = self._inv_boxcox(df[col])

        return df


    def process_output(data, granularity):
        """
        Format data for frontend

        Attributes:
        data: the input dataframe (with anomalies added)
        granularity: data granularity (seconds)

        Returns:
        results: dictionary containing
            y: input timeseries
            yhat_upper: upper confidence bound
            yhat_lower: lower confidence bound
            anomalies: all detected anomalies
        """

        data["ds"] = data["ds"].astype(np.int64) * 1e-9
        anomalies_data = data[~data["anomalies"].isna()][["ds", "anomalies", "y", "yhat"]]
        anomalies = []
        if len(anomalies_data) > 0:
            anomalies = _aggregate_anomalies(anomalies_data, granularity)

        results = {
            "y": _convert_ts(data, "y"),
            "yhat_upper": _convert_ts(data, "yhat_upper"),
            "yhat_lower": _convert_ts(data, "yhat_lower"),
            "anomalies": anomalies,
        }
        return results

    def _boxcox(self, y):
        transformed, self.bc_lambda = stats.boxcox(y + 1)
        if self.bc_lambda <= 0:
            transformed = np.log(y + 1)
        return transformed

    def _inv_boxcox(self, y):
        if self.bc_lambda <= 0:
            return np.exp(y) - 1
        return special.inv_boxcox(y, self.bc_lambda) - 1
    

def _aggregate_anomalies(data, granularity):
    """
    Group consecutive anomalies together into single
    records (with expanded start/end times)

    Attributes:
    data: the input dataframe (with anomalies added)
    granularity: data granularity (in seconds)

    Returns:
    results: list of dictionaries containing combined anomalies
        start: when anomaly started
        end: when anomaly ended
        confidence: anomaly confidence
        received: actual count for metric
        expected: expected count for metric (from yhat)
        id: id/label for each anomaly
    """
    score_map = {1: "low", 2: "high"}
    score_lookup = {v: k for k, v in score_map.items()}
    anomalies = []
    previous_time = None
    anomaly_index = -1
    for ds_time, score, y, yhat in data.itertuples(index=False):
        if previous_time and ds_time <= previous_time + (granularity * 3):
            anomalies[anomaly_index]["end"] = int(ds_time + granularity)
            anomalies[anomaly_index]["received"] += round(y, 5)
            anomalies[anomaly_index]["expected"] += round(yhat, 5)
            anomalies[anomaly_index]["confidence"] = score_map[
                max(score, score_lookup[anomalies[anomaly_index]["confidence"]])
            ]
        else:
            anomaly_index += 1
            anomalies.append(
                {
                    "start": int(ds_time),
                    "end": int(ds_time + granularity),
                    "confidence": score_map[score],
                    "received": round(y, 5),
                    "expected": round(yhat, 5),
                    "id": anomaly_index,
                }
            )
        previous_time = ds_time

    return anomalies


def _convert_ts(ts, value_col):
    """
    Format a timeseries for the frontend
    """
    data = zip(
        list(map(int, ts["ds"])),
        [[{"count": round(x, 5)}] for x in list(ts[value_col])],
    )
    start = int(ts["ds"].iloc[0])
    end = int(ts["ds"].iloc[-1])
    return {"data": list(data), "start": start, "end": end}
