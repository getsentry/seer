import json
import logging
import os
# from fbprophet import Prophet
from kats.models.prophet import ProphetModel, ProphetParams
from kats.detectors.cusum_detection import CUSUMDetector
from kats.consts import TimeSeriesData
from flask import Flask, request
from datetime import datetime
import pandas as pd
import numpy as np

log = logging.getLogger()
app = Flask(__name__)

import time

@app.route("/changepoint", endpoint="changepoint", methods=["POST"])
def changepoint():
    s = time.time()
    data = request.get_json()

    train = pd.DataFrame()
    train["y"] = data["train"]["y"]
    train["ds"] = data["train"]["ds"]
    train["ds"] = pd.to_datetime(train["ds"], format="%Y-%m-%d %H:%M:%S")

    tsd = TimeSeriesData(train, time_col_name="ds")

    detector = CUSUMDetector(tsd)
    change_points = detector.detector(change_directions=["increase"])

    cp, metadata = change_points[0]

    return {"change_start":datetime.strftime(cp._start_time, format="%Y-%m-%d %H:%M:%S")}


@app.route("/predict", endpoint="predict", methods=["POST"])
def predict():
    s = time.time()
    data = request.get_json()

    train = pd.DataFrame()
    train["value"] = data["train"]["value"]
    train["time"] = data["train"]["time"]

    # test = pd.DataFrame()
    # test["y"] = data["test"]["y"]
    # test["ds"] = data["test"]["ds"]
    # y_vals = data["test"]["y"]

    train["time"] = pd.to_datetime(train["time"], format="%Y-%m-%d %H:%M:%S")
    # test["ds"] = pd.to_datetime(test["ds"], format="%Y-%m-%d %H:%M:%S")
    train_ts = TimeSeriesData(train)
    # train["ds"] = pd.to_datetime(train["ds"],unit="s")
    # test["ds"] = pd.to_datetime(test["ds"],unit="s")

    # create a model param instance
    params = ProphetParams(
        seasonality_mode="multiplicative",
        interval_width=0.95,
        daily_seasonality=False,
        weekly_seasonality=40,
        changepoint_range=0.9,
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=0.1
    )


    # create a prophet model instance
    m = ProphetModel(train_ts, params)

    m.fit()

    fcst = m.predict(steps=100)
    add_prophet_uncertainty(p, fcst, using_train_df=True)
    fcst["y"] = y_vals
    print(time.time() - s)

    return fcst[["ds", "yhat_lower", "yhat_upper", "y"]].to_json()


def _make_historical_mat_time(deltas, changepoints_t, t_time, n_row=1):
    """
    Creates a matrix of slope-deltas where these changes occured in training data according to the trained prophet obj
    """
    diff = np.diff(t_time).mean()
    prev_time = np.arange(0, 1 + diff, diff)
    idxs = []
    for changepoint in changepoints_t:
        idxs.append(np.where(prev_time > changepoint)[0][0])
    prev_deltas = np.zeros(len(prev_time))
    prev_deltas[idxs] = deltas
    prev_deltas = np.repeat(prev_deltas.reshape(1, -1), n_row, axis=0)
    return prev_deltas, prev_time


def prophet_logistic_uncertainty(
    mat: np.ndarray,
    deltas: np.ndarray,
    prophet_obj: ProphetModel,
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

    k = prophet_obj.params["k"][0]
    m = prophet_obj.params["m"][0]
    n_length = len(t_time)
    #  for logistic growth we need to evaluate the trend all the way from the start of the train item
    historical_mat, historical_time = _make_historical_mat_time(
        deltas, prophet_obj.changepoints_t, t_time, len(mat)
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
    mean_delta: float, likelihood: float, future_length: float, k: int = 10000
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
    prophet_obj: ProphetModel,
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
    interval_width = prophet_obj.params.interval_width

    if (
        using_train_df
    ):  # there is no trend-based uncertainty if we're only looking on the past where trend is known
        sample_trends = np.zeros((100, len(forecast_df)))
    else:  # create samples of possible future trends
        future_time_series = ((forecast_df["ds"] - prophet_obj.start) / prophet_obj.t_scale).values
        single_diff = np.diff(future_time_series).mean()
        change_likelihood = len(prophet_obj.model.changepoints_t) * single_diff
        deltas = prophet_obj.params["delta"][0]
        n_length = len(forecast_df)
        mean_delta = np.mean(np.abs(deltas)) + 1e-8
        if prophet_obj.growth == "linear":
            mat = _make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=10000)
            sample_trends = mat.cumsum(axis=1).cumsum(axis=1)  # from slope changes to actual values
            sample_trends = sample_trends * single_diff  # scaled by the actual meaning of the slope
        elif prophet_obj.growth == "logistic":
            mat = _make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=1000)
            cap_scaled = (forecast_df["cap"] / prophet_obj.model.y_scale).values
            sample_trends = prophet_logistic_uncertainty(
                mat, deltas, prophet_obj, cap_scaled, future_time_series
            )
        else:
            raise NotImplementedError

    # add gaussian noise based on historical levels
    sigma = prophet_obj.model.params["sigma_obs"][0]
    historical_variance = np.random.normal(scale=sigma, size=sample_trends.shape)
    full_samples = sample_trends + historical_variance
    # get quantiles and scale back (prophet scales the data before fitting, so sigma and deltas are scaled)
    width_split = (1 - interval_width) / 2
    quantiles = np.array([width_split, 1 - width_split]) * 100  # get quantiles from width
    quantiles = np.percentile(full_samples, quantiles, axis=0)
    # Prophet scales all the data before fitting and predicting, y_scale re-scales it to original values
    quantiles = quantiles * prophet_obj.model.y_scale

    forecast_df["yhat_lower"] = forecast_df.yhat + quantiles[0]
    forecast_df["yhat_upper"] = forecast_df.yhat + quantiles[1]
