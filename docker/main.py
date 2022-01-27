import logging
import time

from flask import Flask, request
import pandas as pd
import numpy as np
from datetime import datetime

from prophet_detector import ProphetDetector, ProphetParams

log = logging.getLogger()
app = Flask(__name__)


@app.route("/predict", methods=["GET", "POST"]
def predict():
    timings = {}
    s = time.time()
    if request.method == "GET":
        start, end = args.get("start"), args.get("end")
        query_start, query_end, granularity = map_snuba_queries(start, end)
        data = snuba_query(
            query_start,
            query_end,
            granularity,
            args.get("project"),
            args.get("transaction"),
        )
        start, end = datetime.fromtimestamp(start), datetime.fromtimestamp(end)
    elif request.method == "POST":
        data = request.get_json()
        start, end = data["start"], data["end"]

    params = ProphetParams(
        interval_width=0.95,
        changepoint_prior_scale=0.01,
        weekly_seasonality=14,
        daily_seasonality=False,
        uncertainty_samples=None,
    )
    m = ProphetDetector(pd.DataFrame(data["data"]), start, end, params)
    m.pre_process_data()
    timings["pre_process"] = time.time() - s

    s = time.time()
    m.fit()
    timings["train"] = time.time() - s

    s = time.time()
    fcst = m.predict()
    timings["predict"] = time.time() - s

    s = time.time()
    m.add_prophet_uncertainty(fcst)
    timings["uncertainty"] = time.time() - s

    s = time.time()
    fcst = m.scale_scores(fcst)
    # convert datetime to unix seconds
    fcst["ds"] = fcst["ds"].astype(np.int64) * 1e-9
    timings["gen_scores"] = time.time() - s

    s = time.time()
    output = process_output(fcst, granularity)
    timings["format_output"] = time.time() - s

    logging.info(timings)

    return output


def map_snuba_queries(start, end):
    """
    Takes visualization start/end timestamps
    and returns the start/end/granularity
    of the snuba query that we should execute

    Attributes:
    start: unix timestamp representing start of visualization window
    end: unix timestamp representing end of visualization window

    Returns:
    results: dictionary containing
        query_start: datetime representing start of query window
        query_end: datetime representing end of query window
        granularity: granularity to use (in seconds)
    """

    def days(n):
        return 60 * 60 * 24 * n

    if end - start <= days(2):
        granularity = 300
        query_start = end - days(7)
    elif end - start <= days(7):
        granularity = 600
        query_start = end - days(14)
    elif end - start <= days(14):
        granularity = 1200
        query_start = end - days(28)
    else:
        granularity = 3600
        query_start = end - days(90)
    query_end = end

    return (
        datetime.fromtimestamp(query_start),
        datetime.fromtimestamp(query_end),
        granularity,
    )


def snuba_query(query_start, query_end, granularity, project_id, transaction):
    """
    query_start: starting datetime
    query_end: ending datetime
    granularity: data granularity
    project_id: project_id
    transaction: transaction name
    """
    return None


def aggregate_anomalies(data, granularity):
    """
    Format data for frontend

    Attributes:
    data: the input dataframe (with anomalies added)
    granularity: data granularity (in seconds)

    Returns:
    results: list of dictionaries containing anomaly information
        start: when anomaly started
        end: when anomaly ended
        confidence: anomaly confidence
        received: actual count for metric
        expected: expected count for metric (from yhat)
        id: "unique" id for each anomaly
    """
    anomalies = []
    last_score = None
    anomaly_index = -1
    sum_expected, sum_actual = 0, 0
    for ds_time, score, y, yhat in data[~data["anomalies"].isna()][
        ["ds", "anomalies", "y", "yhat"]
    ].itertuples(index=False):
        if score == last_score:
            anomalies[anomaly_index]["end"] = ds_time + granularity
            anomalies[anomaly_index]["received"] += y
            anomalies[anomaly_index]["expected"] += yhat
        else:
            sum_expected = yhat
            sum_actua = y
            anomaly_index += 1
            anomalies.append(
                {
                    "start": int(ds_time),
                    "end": int(ds_time + granularity),
                    "confidence": score,
                    "received": y,
                    "expected": yhat,
                    "id": anomaly_index,
                }
            )
    last_score = score

    return anomalies


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

    def convert_ts(ts, value_col):
        """
        Format a timeseries for the frontend
        """
        data = zip(
            list(map(int, ts["ds"])), [[{"count": x}] for x in list(ts[value_col])]
        )
        start = int(ts["ds"].iloc[0])
        end = int(ts["ds"].iloc[-1])
        return {"data": list(data), "start": start, "end": end}

    results = {
        "y": convert_ts(data, "y"),
        "yhat_upper": convert_ts(data, "yhat_upper"),
        "yhat_lower": convert_ts(data, "yhat_lower"),
        "anomalies": aggregate_anomalies(data, granularity),
    }
    return results
