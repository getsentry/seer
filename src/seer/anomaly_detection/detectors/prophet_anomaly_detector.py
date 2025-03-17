from datetime import timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd
import sentry_sdk
from prophet import Prophet  # type: ignore
from pydantic import BaseModel
from scipy import special, stats  # type: ignore
from tsmoothie.smoother import SpectralSmoother  # type: ignore

from seer.anomaly_detection.models import AlgoConfig, Sensitivities
from seer.anomaly_detection.models.external import TimePeriods
from seer.dependency_injection import inject, injected


class ProphetAnomalyDetector(BaseModel):
    """
    Anomaly Detection class built around Prophet.

    The class uses a trained prophet model to build confidence intervals
    around a timeseries, and those confidence intervals to establish
    anomaly scores by comparing the intervals to observed values.

    """

    @inject
    @sentry_sdk.trace
    def predict(
        self,
        timestamps: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        forecast_len: int,
        time_period: TimePeriods,
        sensitivity: Sensitivities,
        algo_config: AlgoConfig = injected,
    ) -> pd.DataFrame:
        """
        Predict the future values of the time series

        Args:
            timestamps: timestamps of the time series used for training
            values: values of the time series used for training
            forecast_len: number of future timestamps to forecast
            time_period: granularity of the time series
            sensitivity: sensitivity of the time series
            algo_config: algorithm configuration

        Returns:
            A dataframe with the forecasted values, confidence intervals, and actual values
        """
        df_train = pd.DataFrame({"ds": timestamps, "y": values})
        df_train.ds = pd.to_datetime(df_train.ds, unit="s", utc=True)
        df_train.ds = df_train.ds.dt.tz_localize(None)
        ts_value_map = df_train.set_index("ds")["y"].to_dict()

        df_train.sort_values(by="ds", inplace=True)

        df_train, bc_lambda = self._pre_process_data(df_train, time_period)
        model = self._fit(df_train, sensitivity, algo_config)

        future = model.make_future_dataframe(periods=forecast_len, freq=f"{time_period}min")
        forecast = model.predict(future)
        forecast.ds = pd.to_datetime(forecast.ds)
        forecast.index = forecast["ds"]

        forecast["actual"] = None
        forecast.actual = forecast.ds.map(lambda x: ts_value_map[x] if x in ts_value_map else None)

        forecast.actual = forecast.actual.astype(np.float64)
        forecast["y"] = forecast.actual
        forecast = self._add_prophet_uncertainty(
            forecast, model, sensitivity, algo_config, bc_lambda
        )

        return forecast

    def _pre_process_data(
        self, train: pd.DataFrame, granularity: int
    ) -> tuple[pd.DataFrame, float]:
        """
        Apply kalman filter and log transform input data

        Args:
            train: training dataset
            granularity: granularity of the time series

        Returns:
            A tuple of the training dataset and the box-cox lambda used to undo log transform

        """
        train_df = train.copy()
        train_df["y"] = train_df["y"].fillna(0)
        nans = train_df["y"].isna().sum()
        nans_ds = train_df["ds"].isna().sum()
        infs = train_df["y"].isin([np.inf, -np.inf]).sum()
        if nans > 0 or infs > 0 or nans_ds > 0:
            raise ValueError(
                f"Found {nans} NaNs and {infs} infs in the data. Spectral smoother cannot handle NaNs or infs."
            )
        bc_lambda = 0.0
        # no need to preprocess data if it is constant
        if train_df["y"].nunique() != 1:
            smoother = SpectralSmoother(smooth_fraction=0.35, pad_len=10)
            smoother.smooth(train_df["y"].tolist())
            train_df["y"] = smoother.smooth_data[0]
            train_df["y"] = np.where(smoother.smooth_data[0] < 0, 0, smoother.smooth_data[0])
            train_df["y"], bc_lambda = self._boxcox(train_df["y"])

        train_df["y"] = train_df["y"][: train.shape[0]]

        # we are using zerofill=True, so we need to fill in records even if there is no data
        train_df = train_df.set_index("ds", drop=False).asfreq(timedelta(minutes=granularity))
        train_df["ds"] = train_df.index
        # train_df.reset_index(drop=True, inplace=True)
        return train_df, bc_lambda

    def _fit(
        self,
        df: pd.DataFrame,
        sensitivity: Sensitivities,
        algo_config: AlgoConfig,
    ) -> Prophet:
        """
        Fit Prophet model

        Args:
            df: DataFrame with training data
            sensitivity: Sensitivity level
            algo_config: Algorithm configuration

        Returns:
            The fitted prophet model object
        """
        model_params = algo_config.get_prophet_params(sensitivity)

        prophet = Prophet(
            growth=model_params.growth,
            changepoints=model_params.changepoints,
            n_changepoints=model_params.n_changepoints,
            changepoint_range=model_params.changepoint_range,
            yearly_seasonality=model_params.yearly_seasonality,
            weekly_seasonality=model_params.weekly_seasonality,
            daily_seasonality=model_params.daily_seasonality,
            holidays=model_params.holidays,
            seasonality_mode=model_params.seasonality_mode,
            seasonality_prior_scale=model_params.seasonality_prior_scale,
            holidays_prior_scale=model_params.holidays_prior_scale,
            changepoint_prior_scale=model_params.changepoint_prior_scale,
            mcmc_samples=model_params.mcmc_samples,
            interval_width=model_params.interval_width,
            uncertainty_samples=model_params.uncertainty_samples,
        )

        if model_params.growth == "logistic":
            # assign cap to a new col as Prophet required
            df["cap"] = model_params.cap

        # Adding floor if available
        if model_params.floor is not None:
            df["floor"] = model_params.floor

        # Add any specified custom seasonalities.
        for custom_seasonality in model_params.custom_seasonalities:
            prophet.add_seasonality(**custom_seasonality)

        # limit iter to 250 to avoid long inference times
        model = prophet.fit(df=df, iter=250)
        return model

    def _add_prophet_uncertainty(
        self,
        df: pd.DataFrame,
        model: Prophet,
        sensitivity: Sensitivities,
        algo_config: AlgoConfig,
        bc_lambda: float,
    ) -> pd.DataFrame:
        """
        Adds yhat_upper and yhat_lower to the forecast_df,
        based on the params of a trained prophet model and the interval_width.

        Args:
            df: DataFrame with predicted values
            model: trained prophet model
            sensitivity: sensitivity level
            algo_config: algorithm configuration
            bc_lambda: box-cox lambda used to undo log transform

        Returns:
            DataFrame with confidence intervals (yhat_upper and yhat_lower) added
        """
        assert "yhat" in df.columns, "Must have the mean yhat forecast to build uncertainty on"
        interval_width = algo_config.get_prophet_params(sensitivity).interval_width

        # there is no trend-based uncertainty if we're only looking on the past where trend is known
        sample_trends = np.zeros((2000, len(df)))

        # add gaussian noise based on historical levels
        sigma = model.params["sigma_obs"][0]
        historical_variance = np.random.normal(scale=sigma, size=sample_trends.shape)
        full_samples = sample_trends + historical_variance
        # get quantiles and scale back (prophet scales the data before fitting, so sigma and deltas are scaled)
        width_split = (1 - interval_width) / 2
        quantiles = np.array([width_split, 1 - width_split]) * 100  # get quantiles from width
        quantiles = np.percentile(full_samples, quantiles, axis=0)
        # Prophet scales all the data before fitting and predicting, y_scale re-scales it to original values
        quantiles = quantiles * model.y_scale

        df["yhat_lower"] = quantiles[0] + df.yhat
        df["yhat_upper"] = quantiles[1] + df.yhat

        should_invert = True if df["actual"].nunique() != 1 else False
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            df[col] = np.where(df[col] < 0.0, 0.0, df[col])
            if should_invert:
                df[col] = self._inv_boxcox(df[col], bc_lambda)

        return df

    def _boxcox(self, y) -> tuple[npt.NDArray[np.float64], float]:
        transformed, bc_lambda = stats.boxcox(y + 1)
        if bc_lambda <= 0:
            transformed = np.log(y + 1)
        return transformed, bc_lambda

    def _inv_boxcox(self, y, bc_lambda) -> npt.NDArray[np.float64]:
        if bc_lambda <= 0:
            return np.exp(y) - 1
        return special.inv_boxcox(y, bc_lambda) - 1
