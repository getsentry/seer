import sentry_sdk
import time
import os

from flask import Flask, request
from sentry_sdk.integrations.flask import FlaskIntegration
import pandas as pd
import numpy as np
from datetime import datetime

from prophet_detector import ProphetDetector, ProphetParams

sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0,
)
app = Flask(__name__)

MODEL_PARAMS = ProphetParams(
    interval_width=0.95,
    changepoint_prior_scale=0.01,
    weekly_seasonality=14,
    daily_seasonality=False,
    uncertainty_samples=None,
)

@app.route("/anomaly/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        start, end = args.get("start"), args.get("end")
        query_start, query_end, granularity = map_snuba_queries(start, end)
        with sentry_sdk.start_span(op="snuba.query", description="Query anomaly training dataset from snuba") as span:
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
        granularity = data["granularity"]

    with sentry_sdk.start_span(op="data.preprocess", description="Preprocess data to prepare for anomaly detection") as span:
        m = ProphetDetector(start, end, MODEL_PARAMS)
        m.pre_process_data(pd.DataFrame(data["data"]))

    with sentry_sdk.start_span(op="model.train", description="Train forecasting model") as span:
        m.fit()

    with sentry_sdk.start_span(op="model.predict", description="Generate predictions") as span:
        fcst = m.predict()

    with sentry_sdk.start_span(op="model.confidence", description="Generate confidence intervals") as span:
        m.add_prophet_uncertainty(fcst)

    with sentry_sdk.start_span(op="data.anomaly.scores", description="Generate anomaly scores using forecast") as span:
        fcst = m.scale_scores(fcst)

    with sentry_sdk.start_span(op="data.format", description="Format data for frontend") as span:
        fcst["ds"] = fcst["ds"].astype(np.int64) * 1e-9
        output = process_output(fcst, granularity)

    return output

@app.route("/health/live", methods=["GET"])
def health_check():
    return "OK"

@app.route("/health/ready", methods=["GET"])
def ready_check():
    m = ProphetDetector("2022-01-01", "2022-01-14", MODEL_PARAMS)
    return "OK"


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
    for ds_time, score, y, yhat in data.itertuples(index=False):
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

    anomalies_data = data[~data["anomalies"].isna()][["ds", "anomalies", "y", "yhat"]]
    anomalies = []
    if len(anomalies_data) > 0:
        anomalies = aggregate_anomalies(anomalies_data, granularity)

    results = {
        "y": convert_ts(data, "y"),
        "yhat_upper": convert_ts(data, "yhat_upper"),
        "yhat_lower": convert_ts(data, "yhat_lower"),
        "anomalies": anomalies,
        "scores": convert_ts(data, "final_score")
    }
    return results
