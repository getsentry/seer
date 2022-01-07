import logging

from prophet_model import ProphetModel, ProphetParams
from kats.detectors.cusum_detection import CUSUMDetector
from kats.consts import TimeSeriesData

from flask import Flask, request
from datetime import datetime
import pandas as pd

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

    return dict(change_start=datetime.strftime(cp._start_time, format="%Y-%m-%d %H:%M:%S"))


@app.route("/predict", endpoint="predict", methods=["POST"])
def predict():

    timings = {}
    s = time.time()
    data = request.get_json()
    start, end = data["start"], data["end"]
    params = ProphetParams(
        interval_width=0.95,
        changepoint_prior_scale=0.01,
        weekly_seasonality=14,
        daily_seasonality=False,
        uncertainty_samples=None
    )
    m = ProphetModel(data["train"], start, end, params)

    timings["pre_process"] = time.time() - s

    s = time.time()
    m.fit()
    timings["train"] = time.time() - s

    s = time.time()
    fcst = m.gen_forecast()
    timings["predict"] = time.time() - s
    s = time.time()
    m.add_prophet_uncertainty(fcst, using_train_df=True)
    timings["uncertainty"] = time.time() - s
    s = time.time()
    fcst = m.gen_scores(fcst)
    timings["gen_scores"] = time.time() - s
    print(timings)

    return fcst[["ds", "yhat_lower", "yhat_upper", "y", "score", "scaled_score", "final_score"]].to_json()
