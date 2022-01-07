import logging
from typing import Dict, List, Optional

import kats.models.model as m
import pandas as pd
import numpy as np
from fbprophet import Prophet
from kats.consts import Params, TimeSeriesData
from tsmoothie.smoother import SpectralSmoother
from scipy import stats

class ProphetParams(Params):
    """Parameter class for Prophet model

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
        custom_seasonalities: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__()
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
            [] if custom_seasonalities is None
            else custom_seasonalities
        )
        logging.debug(
            "Initialized Prophet with parameters. "
            "growth:{growth},"
            "changepoints:{changepoints},"
            "n_changepoints:{n_changepoints},"
            "changepoint_range:{changepoint_range},"
            "yearly_seasonality:{yearly_seasonality},"
            "weekly_seasonality:{weekly_seasonality},"
            "daily_seasonality:{daily_seasonality},"
            "holidays:{holidays},"
            "seasonality_mode:{seasonality_mode},"
            "seasonality_prior_scale:{seasonality_prior_scale},"
            "holidays_prior_scale:{holidays_prior_scale},"
            "changepoint_prior_scale:{changepoint_prior_scale},"
            "mcmc_samples:{mcmc_samples},"
            "interval_width:{interval_width},"
            "uncertainty_samples:{uncertainty_samples},"
            "cap:{cap},"
            "floor:{floor},"
            "custom_seasonalities:{custom_seasonalities}".format(
                growth=growth,
                changepoints=changepoints,
                n_changepoints=n_changepoints,
                changepoint_range=changepoint_range,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                holidays=holidays,
                seasonality_mode=seasonality_mode,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                changepoint_prior_scale=changepoint_prior_scale,
                mcmc_samples=mcmc_samples,
                interval_width=interval_width,
                uncertainty_samples=uncertainty_samples,
                cap=cap,
                floor=floor,
                custom_seasonalities=custom_seasonalities,
            )
        )

    def validate_params(self):
        """validate Prophet parameters

        This method validates some key parameters including growth rate
        and custom_seasonalities.
        """
        # cap must be given when using logistic growth
        if (self.growth == "logistic") and (self.cap is None):
            msg = "Capacity must be provided for logistic growth"
            logging.error(msg)
            raise ValueError(msg)

        # If custom_seasonalities passed, ensure they contain the required keys.
        reqd_seasonality_keys = ["name", "period", "fourier_order"]
        if not all(
            req_key in seasonality
            for req_key in reqd_seasonality_keys
            for seasonality in self.custom_seasonalities
        ):
            msg = f"Custom seasonality dicts must contain the following keys:\n{reqd_seasonality_keys}"
            logging.error(msg)
            raise ValueError(msg)

        logging.info("Method validate_params() is not fully implemented.")
        pass



class ProphetModel(m.Model):
    """Model class for Prophet

    This class provides fit, predict, and plot methods for Prophet model

    Attributes:
        data: the input time series data as in :class:`kats.consts.TimeSeriesData`
        params: the parameter class definied with `ProphetParams`
    """
    def __init__(self, data: pd.DataFrame, start, end, params: ProphetParams) -> None:
        train, self.test, self.bc_lambda = self.pre_process_data(data)
        self.start, self.end = start, end
        super().__init__(train, params)
        if not isinstance(self.data.value, pd.Series):
            msg = "Only support univariate time series, but get {type}.".format(
                type=type(self.data.value)
            )
            logging.error(msg)
            raise ValueError(msg)

    def pre_process_data(self, data):
        """
        Apply kalman filter and log transform
        """
        data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d %H:%M:%S")
        data.index = data["time"]

        smoother = SpectralSmoother(smooth_fraction=0.25, pad_len=10)
        smoother.smooth(list(data["value"]))
        data["value"] = smoother.smooth_data[0]
        data["value"] = np.where(smoother.smooth_data[0] < 0, 0, smoother.smooth_data[0])

        data["value"], bc_lambda = stats.boxcox(data["value"] + 1)
        test = data[self.start:self.end]

        return TimeSeriesData(data), test, bc_lambda

    def fit(self, **kwargs) -> None:
        """fit Prophet model

        Args:
            None.

        Returns:
            The fitted prophet model object
        """
        # prepare dataframe for Prophet.fit()
        df = pd.DataFrame({"ds": self.data.time, "y": self.data.value})
        logging.debug(
            "Call fit() with parameters: "
            "growth:{growth},"
            "changepoints:{changepoints},"
            "n_changepoints:{n_changepoints},"
            "changepoint_range:{changepoint_range},"
            "yearly_seasonality:{yearly_seasonality},"
            "weekly_seasonality:{weekly_seasonality},"
            "daily_seasonality:{daily_seasonality},"
            "holidays:{holidays},"
            "seasonality_mode:{seasonality_mode},"
            "seasonality_prior_scale:{seasonality_prior_scale},"
            "holidays_prior_scale:{holidays_prior_scale},"
            "changepoint_prior_scale:{changepoint_prior_scale},"
            "mcmc_samples:{mcmc_samples},"
            "interval_width:{interval_width},"
            "uncertainty_samples:{uncertainty_samples},"
            "cap:{cap},"
            "floor:{floor},"
            "custom_seasonalities:{custom_seasonalities}".format(
                growth=self.params.growth,
                changepoints=self.params.changepoints,
                n_changepoints=self.params.n_changepoints,
                changepoint_range=self.params.changepoint_range,
                yearly_seasonality=self.params.yearly_seasonality,
                weekly_seasonality=self.params.weekly_seasonality,
                daily_seasonality=self.params.daily_seasonality,
                holidays=self.params.holidays,
                seasonality_mode=self.params.seasonality_mode,
                seasonality_prior_scale=self.params.seasonality_prior_scale,
                holidays_prior_scale=self.params.holidays_prior_scale,
                changepoint_prior_scale=self.params.changepoint_prior_scale,
                mcmc_samples=self.params.mcmc_samples,
                interval_width=self.params.interval_width,
                uncertainty_samples=self.params.uncertainty_samples,
                cap=self.params.cap,
                floor=self.params.floor,
                custom_seasonalities=self.params.custom_seasonalities,
            )
        )

        prophet = Prophet(
            growth=self.params.growth,
            changepoints=self.params.changepoints,
            n_changepoints=self.params.n_changepoints,
            changepoint_range=self.params.changepoint_range,
            yearly_seasonality=self.params.yearly_seasonality,
            weekly_seasonality=self.params.weekly_seasonality,
            daily_seasonality=self.params.daily_seasonality,
            holidays=self.params.holidays,
            seasonality_mode=self.params.seasonality_mode,
            seasonality_prior_scale=self.params.seasonality_prior_scale,
            holidays_prior_scale=self.params.holidays_prior_scale,
            changepoint_prior_scale=self.params.changepoint_prior_scale,
            mcmc_samples=self.params.mcmc_samples,
            interval_width=self.params.interval_width,
            uncertainty_samples=self.params.uncertainty_samples,
        )

        if self.params.growth == "logistic":
            # assign cap to a new col as Prophet required
            df["cap"] = self.params.cap

        # Adding floor if available
        if self.params.floor is not None:
            df["floor"] = self.params.floor

        # Add any specified custom seasonalities.
        for custom_seasonality in self.params.custom_seasonalities:
            prophet.add_seasonality(**custom_seasonality)

        # pyre-fixme[16]: `ProphetModel` has no attribute `model`.
        self.model = prophet.fit(df=df, iter=250)
        logging.info("Fitted Prophet model. ")


    # pyre-fixme[14]: `predict` overrides method defined in `Model` inconsistently.
    def predict(self, steps, include_history=False, **kwargs) -> pd.DataFrame:
        """predict with fitted Prophet model

        Args:
            steps: the steps or length of prediction horizon
            include_history: if include the historical data, default as False

        Returns:
            The predicted dataframe with following columns:
                `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        logging.debug(
            "Call predict() with parameters. "
            "steps:{steps}, kwargs:{kwargs}".format(steps=steps, kwargs=kwargs)
        )
        # pyre-fixme[16]: `ProphetModel` has no attribute `freq`.
        # pyre-fixme[16]: `ProphetModel` has no attribute `data`.
        self.freq = kwargs.get("freq", pd.infer_freq(self.data.time))
        # pyre-fixme[16]: `ProphetModel` has no attribute `include_history`.
        self.include_history = include_history
        # prepare future for Prophet.predict
        future = kwargs.get("future")
        raw = kwargs.get("raw", False)
        if future is None:
            # pyre-fixme[16]: `ProphetModel` has no attribute `model`.
            # pyre-fixme[16]: `Params` has no attribute `cap`.
            future = self.model.make_future_dataframe(
                periods=steps,
                freq=self.freq,
                include_history=self.include_history)
            if self.params.growth == "logistic":
                # assign cap to a new col as Prophet required
                future["cap"] = self.params.cap
            if self.params.floor is not None:
                future["floor"] = self.params.floor

        fcst = self.model.predict(future).tail(steps)
        if raw:
            return fcst

        # if include_history:
        fcst = self.model.predict(future)
        logging.info("Generated forecast data from Prophet model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        # pyre-fixme[16]: `ProphetModel` has no attribute `fcst_df`.
        self.fcst_df = pd.DataFrame(
            {
                "time": fcst.ds,
                "fcst": fcst.yhat,
                "fcst_lower": fcst.yhat_lower,
                "fcst_upper": fcst.yhat_upper,
            }
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return self.fcst_df

    def gen_forecast(self):
        forecast = self.model.predict(self.test)
        forecast['y'] = self.test['y'].values

        return forecast

    def _inv_box(self, y):
        if self.bc_lambda == 0:
            return np.exp(y) - 1
        else:
            return np.exp(np.log(self.bc_lambda * y + 1) / self.bc_lambda) - 1

    def gen_scores(self, forecast):
        forecast["y"] = self._inv_box(forecast["y"])
        forecast["yhat"] = self._inv_box(forecast["yhat"])
        forecast['yhat_upper'] = self._inv_box(forecast["yhat_upper"])
        forecast['yhat_lower'] = self._inv_box(forecast["yhat_lower"])

        forecast["yhat_lower"] = np.where(forecast["yhat_lower"] < 0, 0, forecast["yhat_lower"])
        forecast['score'] = (
                    (forecast['y'] - forecast['yhat_upper']) * (forecast['y'] >= forecast['yhat']) +
                    (forecast['yhat_lower'] - forecast['y']) * (forecast['y'] < forecast['yhat'])
            )

        pos_score_max = max(forecast["score"])
        pos_score_min = min(forecast[forecast["score"] > 0]["score"])
        neg_score_max = max(forecast[forecast["score"] < 0]["score"])
        neg_score_min = min(forecast["score"])

        forecast["scaled_score"] = np.where(
            forecast["score"] > 0,
            (forecast["score"] - pos_score_min) / pos_score_max,
            (forecast["score"] - neg_score_max) / (-1 * neg_score_min)
        )
        forecast["final_score"] = self.scale_anomaly_data(forecast["scaled_score"], 6)

        return forecast

    def _make_historical_mat_time(self, deltas, t_time, n_row=1):
        """
        Creates a matrix of slope-deltas where these changes occured in training data according to the trained prophet obj
        """
        diff = np.diff(t_time).mean()
        prev_time = np.arange(0, 1 + diff, diff)
        idxs = []
        for changepoint in self.model.changepoints_t:
            idxs.append(np.where(prev_time > changepoint)[0][0])
        prev_deltas = np.zeros(len(prev_time))
        prev_deltas[idxs] = deltas
        prev_deltas = np.repeat(prev_deltas.reshape(1, -1), n_row, axis=0)
        return prev_deltas, prev_time


    def prophet_logistic_uncertainty(
        self,
        mat: np.ndarray,
        deltas: np.ndarray,
        cap_scaled: np.ndarray,
        t_time: np.ndarray,
    ):
        """
        Vectorizes prophet's logistic growth uncertainty by creating a matrix of future possible trends.
        """

        def ffill(arr):
            mask = arr == 0
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            return arr[np.arange(idx.shape[0])[:, None], idx]

        k = self.params["k"][0]
        m = self.params["m"][0]
        n_length = len(t_time)
        #  for logistic growth we need to evaluate the trend all the way from the start of the train item
        historical_mat, historical_time = self._make_historical_mat_time(
            deltas, t_time, len(mat)
        )
        mat = np.concatenate([historical_mat, mat], axis=1)
        full_t_time = np.concatenate([historical_time, t_time])

        #  apply logistic growth logic on the slope changes
        k_cum = np.concatenate(
            (np.ones((mat.shape[0], 1)) * k, np.where(mat, np.cumsum(mat, axis=1) + k, 0)), axis=1
        )
        k_cum_b = ffill(k_cum)
        gammas = np.zeros_like(mat)
        for i in range(mat.shape[1]):
            x = full_t_time[i] - m - np.sum(gammas[:, :i], axis=1)
            ks = 1 - k_cum_b[:, i] / k_cum_b[:, i + 1]
            gammas[:, i] = x * ks
        # the data before the -n_length is the historical values, which are not needed, so cut the last n_length
        k_t = (mat.cumsum(axis=1) + k)[:, -n_length:]
        m_t = (gammas.cumsum(axis=1) + m)[:, -n_length:]
        sample_trends = cap_scaled / (1 + np.exp(-k_t * (t_time - m_t)))
        # remove the mean because we only need width of the uncertainty centered around 0
        # we will add the width to the main forecast - yhat (which is the mean) - later
        sample_trends = sample_trends - sample_trends.mean(axis=0)
        return sample_trends


    def _make_trend_shift_matrix(
        self, mean_delta: float, likelihood: float, future_length: float, k: int = 10000
    ) -> np.ndarray:
        """
        Creates a matrix of random trend shifts based on historical likelihood and size of shifts.
        Can be used for either linear or logistic trend shifts.
        Each row represents a different sample of a possible future, and each column is a time step into the future.
        """
        # create a bool matrix of where these trend shifts should go
        bool_slope_change = np.random.uniform(size=(k, future_length)) < likelihood
        shift_values = np.random.laplace(0, mean_delta, size=bool_slope_change.shape)
        mat = shift_values * bool_slope_change
        n_mat = np.hstack([np.zeros((len(mat), 1)), mat])[:, :-1]
        mat = (n_mat + mat) / 2
        return mat


    def add_prophet_uncertainty(
        self,
        forecast_df: pd.DataFrame,
        using_train_df: bool = False,
    ):
        """
        Adds yhat_upper and yhat_lower to the forecast_df used by fbprophet, based on the params of a trained prophet_obj
        and the interval_width.
        Use using_train_df=True if the forecast_df is not for a future time but for the training data.
        """
    #     assert prophet_obj.history is not None, "Model has not been fit"
        assert "yhat" in forecast_df.columns, "Must have the mean yhat forecast to build uncertainty on"
        interval_width = self.params.interval_width

        if (
            using_train_df
        ):  # there is no trend-based uncertainty if we're only looking on the past where trend is known
            sample_trends = np.zeros((1000, len(forecast_df)))
        else:  # create samples of possible future trends
            future_time_series = ((forecast_df["ds"] - self.model.start) / self.model.t_scale).values
            single_diff = np.diff(future_time_series).mean()
            change_likelihood = len(self.model.changepoints_t) * single_diff
            deltas = self.params["delta"][0]
            n_length = len(forecast_df)
            mean_delta = np.mean(np.abs(deltas)) + 1e-8
            if self.model.growth == "linear":
                mat = self._make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=10000)
                sample_trends = mat.cumsum(axis=1).cumsum(axis=1)  # from slope changes to actual values
                sample_trends = sample_trends * single_diff  # scaled by the actual meaning of the slope
            elif self.model.growth == "logistic":
                mat = self._make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=1000)
                cap_scaled = (forecast_df["cap"] / self.model.y_scale).values
                sample_trends = self.prophet_logistic_uncertainty(
                    mat, deltas, cap_scaled, future_time_series
                )
            else:
                raise NotImplementedError

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

        forecast_df["yhat_lower"] = forecast_df.yhat + quantiles[0]
        forecast_df["yhat_upper"] = forecast_df.yhat + quantiles[1]

    @staticmethod
    def scale_anomaly_data(data, smoothing):
        history = []
        results = data.copy()
        for i, entry in enumerate(data):
            history.insert(0, entry)
            if len(history) >= smoothing:
                history.pop()
            results[i] = 1 + np.sum(history)
        return results
