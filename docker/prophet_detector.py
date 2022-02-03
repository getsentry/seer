import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from prophet import Prophet
from tsmoothie.smoother import SpectralSmoother
from scipy import stats
from datetime import datetime


class ProphetParams:
    """
    Parameter class for Prophet model

    This is the parameter class for prophet model, it contains all necessary
    parameters as definied in Prophet implementation:
    https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py

    Attributes:
        growth: String 'linear' or 'logistic' to specify a linear or logistic
            trend.
        changepoints: List of dates at which to include potential changepoints. If
            not specified, potential changepoints are selected automatically.
        n_changepoints: Number of potential changepoints to include. Not used
            if input `changepoints` is supplied. If `changepoints` is not supplied,
            then n_changepoints potential changepoints are selected uniformly from
            the first `changepoint_range` proportion of the history.
        changepoint_range: Proportion of history in which trend changepoints will
            be estimated. Defaults to 0.8 for the first 80%. Not used if
            `changepoints` is specified.
        yearly_seasonality: Fit yearly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        weekly_seasonality: Fit weekly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        daily_seasonality: Fit daily seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        holidays: pd.DataFrame with columns holiday (string) and ds (date type)
            and optionally columns lower_window and upper_window which specify a
            range of days around the date to be included as holidays.
            lower_window=-2 will include 2 days prior to the date as holidays. Also
            optionally can have a column prior_scale specifying the prior scale for
            that holiday.
        seasonality_mode: 'additive' (default) or 'multiplicative'.
        seasonality_prior_scale: Parameter modulating the strength of the
            seasonality model. Larger values allow the model to fit larger seasonal
            fluctuations, smaller values dampen the seasonality. Can be specified
            for individual seasonalities using add_seasonality.
        holidays_prior_scale: Parameter modulating the strength of the holiday
            components model, unless overridden in the holidays input.
        changepoint_prior_scale: Parameter modulating the flexibility of the
            automatic changepoint selection. Large values will allow many
            changepoints, small values will allow few changepoints.
        mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
            with the specified number of MCMC samples. If 0, will do MAP
            estimation.
        interval_width: Float, width of the uncertainty intervals provided
            for the forecast. If mcmc_samples=0, this will be only the uncertainty
            in the trend using the MAP estimate of the extrapolated generative
            model. If mcmc.samples>0, this will be integrated over all model
            parameters, which will include uncertainty in seasonality.
        uncertainty_samples: Number of simulated draws used to estimate
            uncertainty intervals. Settings this value to 0 or False will disable
            uncertainty estimation and speed up the calculation.
        cap: capacity, provided for logistic growth
        floor: floor, the fcst value must be greater than the specified floor
    """

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000,
        cap=None,
        floor=None,
        custom_seasonalities: List[Dict] = None,
    ) -> None:
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.cap = cap
        self.floor = floor
        self.custom_seasonalities = (
            [] if custom_seasonalities is None else custom_seasonalities
        )


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

    def pre_process_data(self, data: pd.DataFrame, start: str, end: str):
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
        train = data.rename(columns={"time": "ds", "event_count": "y"})
        train["ds"] = pd.to_datetime(train["ds"], format="%Y-%m-%d %H:%M:%S")
        train.index = train["ds"]

        smoother = SpectralSmoother(smooth_fraction=0.25, pad_len=10)
        smoother.smooth(list(train["y"]))
        train["y"] = smoother.smooth_data[0]
        train["y"] = np.where(smoother.smooth_data[0] < 0, 0, smoother.smooth_data[0])

        train["y"], bc_lambda = stats.boxcox(train["y"] + 1)
        self.test = train[start:end]
        self.train = train
        self.bc_lambda = bc_lambda

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
        forecast["y"] = self.test["y"].values

        return forecast

    def _inv_box(self, y):
        """
        Inverse the box-cox log transform

        Args:
            y: value to be transformed

        Returns:
            Transformed value (undo log transform)
        """
        if self.bc_lambda == 0:
            return np.exp(y) - 1
        else:
            return np.exp(np.log(self.bc_lambda * y + 1) / self.bc_lambda) - 1

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
        # for col in ["y", "yhat", "yhat_lower", "yhat_upper"]:
        #     df[col] = self._inv_box(df[col])

        df["yhat_lower"] = np.where(df["yhat_lower"] < 0, 0, df["yhat_lower"])

        df["score"] = (
            np.where(
                df["y"] >= df["yhat"], df["y"] - df["yhat_upper"], df["yhat_lower"] - df["y"]
            )
            / df["y"].std()
        )
        df["final_score"] = df["score"].rolling(10, center=True, min_periods=1).mean()
        df["anomalies"] = np.where(
            df["final_score"] >= self.high_threshold,
            "high",
            np.where(df["final_score"] >= self.low_threshold, "low", None),
        )

        return df

    def add_prophet_uncertainty(self, df: pd.DataFrame):
        """
        Adds yhat_upper and yhat_lower to the forecast_df,
        based on the params of a trained prophet model and the interval_width.

        Args:
            df: DataFrame with predicted values

        Returns:
            DataFrame with confidence intervals (yhat_upper and yhat_lower) added
        """
        assert (
            "yhat" in df.columns
        ), "Must have the mean yhat forecast to build uncertainty on"
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

        df["yhat_lower"] = df.yhat + quantiles[0]
        df["yhat_upper"] = df.yhat + quantiles[1]

        return df
