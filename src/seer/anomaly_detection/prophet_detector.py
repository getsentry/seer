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
        train["ds"] = pd.to_datetime(train["ds"], unit="s")

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

        for col in ["y", "yhat", "yhat_lower", "yhat_upper"]:
            df[col] = np.where(df[col] < 0.0, 0.0, df[col])
            df[col] = self._inv_boxcox(df[col])

        return df

    def _boxcox(self, y):
        transformed, self.bc_lambda = stats.boxcox(y + 1)
        if self.bc_lambda <= 0:
            transformed = np.log(y + 1)
        return transformed

    def _inv_boxcox(self, y):
        if self.bc_lambda <= 0:
            return np.exp(y) - 1
        return special.inv_boxcox(y, self.bc_lambda) - 1
