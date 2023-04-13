import sentry_sdk
import os

from flask import Flask, request
from sentry_sdk.integrations.flask import FlaskIntegration
import pandas as pd
import numpy as np

from seer.trend_detection.detectors.cusum_detection import CUSUMDetector

from seer.anomaly_detection.prophet_detector import ProphetDetector
from seer.anomaly_detection.prophet_params import ProphetParams

sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0,
)
app = Flask(__name__)

MODEL_PARAMS = ProphetParams(
    interval_width=0.975,
    changepoint_prior_scale=0.01,
    weekly_seasonality=14,
    daily_seasonality=False,
    uncertainty_samples=None,
)
model_initialized = False
detector = ProphetDetector(MODEL_PARAMS)
model_initialized = True


@app.route("/trends/mock", methods=["POST"])
def mock_trends_endpoint():
    data = request.get_json()

    return {
        "events": {
            "data": [{
            "project": "sentry",
            "transaction": "sentry.tasks.check_auth_identity",
            "aggregate_range_1": 11,
            "aggregate_range_2": 64,
            "count_range_1": 13858,
            "count_range_2": 11750,
            "t_test": -13.800140125042828,
            "trend_percentage": 5.818181818181818,
            "trend_difference": 53,
            "count_percentage": 0.8478856977918892,
            "tpm": 1.2702380952380952,
			"breakpoint": int((data['start'] + data['end'])/2) #returns midpoint of time frame for now
            }]
        },

        #or return data back to server?
        "stats": {
            "transaction name": {
                "data": data['data'],
                "start": data['start'],
                "end": data['end'],
                'isMetricsData': True,
                'order': 3,
            }
        }
    }


@app.route("/trends/breakpoint-detector", methods=["POST"])
def breakpoint_trends_endpoint():

    def get_agg_range(seq_data, function):
        if len(seq_data) == 0:
            return 0

        if function == "p50":
            return np.percentile(seq_data, 50)
        if function == "p75":
            return np.percentile(seq_data, 75)
        if function == "p95":
            return np.percentile(seq_data, 95)

        return np.percentile(seq_data, 99)


    data = request.get_json()
    txns_data = data['data']

    output_dict = {}

    for txn in txns_data.keys():

        ts_data = txns_data[txn]['data']

        timestamps = [x[0] for x in ts_data]
        counts = [x[1][0]['count'] for x in ts_data]

        timeseries = pd.DataFrame(
            {
                'time': timestamps,
                'y': counts,
            }
        )

        change_points = CUSUMDetector(timeseries).detector()

        if len(change_points) == 0:
            # use middle of timeseries as breakpoint
            change_point = int((data['start'] + data['end']) / 2)
        else:
            # get most recent change point
            change_point = change_points[-1].start_time

        change_index = timestamps.index(change_point)
        first_half, second_half = counts[:change_index], counts[change_index:]

        trend_function = data['trendFunction'].split("(")[0]

        mu0 = sum(first_half) / len(first_half)
        mu1 = sum(second_half) / len(second_half)

        if trend_function == "avg":
            agg_range_1 = mu0
            agg_range_2 = mu1
        else:
            agg_range_1 = get_agg_range(first_half, trend_function)
            agg_range_2 = get_agg_range(second_half, trend_function)

        count_range_1 = len(first_half)
        count_range_2 = len(second_half)

        # calculate variance of both groups
        var1 = sum((x-agg_range_1)**2 for x in first_half) / count_range_1
        var2 = sum((x-agg_range_2)**2 for x in second_half) / count_range_2

        # calculate t-value between both groups
        t_value = (mu0-mu1) / ((var1/count_range_1) + (var2/count_range_2)) ** (1/2)
        trend_percentage = int(((agg_range_2-agg_range_1)/agg_range_1) * 100)

        output_dict[txn] = {
        "events": {
            "data": [{
            "project": "sentry",
            "transaction": "sentry.tasks.check_auth_identity",
            "aggregate_range_1": agg_range_1,
            "aggregate_range_2": agg_range_2,
            "count_range_1": count_range_1,
            "count_range_2": count_range_2,
            "t_test": t_value,
            "trend_percentage": trend_percentage,
            "trend_difference": agg_range_2 - agg_range_1,
            "count_percentage": count_range_2/count_range_1,
			"breakpoint": change_point
            }]
        },

        # return data back to server?
        "stats": {
            "transaction name": {
                "data": data['data'],
                "start": data['start'],
                "end": data['end'],
                'isMetricsData': True,
            }
        }
        }

    return output_dict


@app.route("/anomaly/predict", methods=["POST"])
def predict():
    data = request.get_json()
    start, end = data.get("start", None), data.get("end", None)
    granularity = data.get("granularity", None)
    ads_context = {
        "detection_window_start": start,
        "detection_window_end": end,
        "low_threshold": detector.low_threshold,
        "high_threshold": detector.high_threshold,
        "interval_width": MODEL_PARAMS.interval_width,
        "changepoint_prior_scale": MODEL_PARAMS.changepoint_prior_scale,
        "weekly_seasonality": MODEL_PARAMS.weekly_seasonality,
        "daily_seasonality": MODEL_PARAMS.daily_seasonality,
        "uncertainty_samples": MODEL_PARAMS.uncertainty_samples,
    }
    snuba_context = {
        "granularity": granularity,
        "query": data.get("query", None),
    }
    sentry_sdk.set_context("snuba_query", snuba_context)
    sentry_sdk.set_context("anomaly_detection_params", ads_context)

    with sentry_sdk.start_span(
        op="data.preprocess", description="Preprocess data to prepare for anomaly detection"
    ) as span:
        if (
            "data" not in data
            or len(data["data"]) == 0
            or not all(key in data["data"][0] for key in ("time", "count"))
        ):
            return {
                "y": {"data": []},
                "yhat_upper": {"data": []},
                "yhat_lower": {"data": []},
                "anomalies": [],
            }
        detector.pre_process_data(pd.DataFrame(data["data"]), granularity, start, end)
        ads_context["boxcox_lambda"] = detector.bc_lambda

    with sentry_sdk.start_span(
        op="model.train", description="Train forecasting model"
    ) as span:
        detector.fit()

    with sentry_sdk.start_span(op="model.predict", description="Generate predictions") as span:
        fcst = detector.predict()

    with sentry_sdk.start_span(
        op="model.confidence", description="Generate confidence intervals"
    ) as span:
        detector.add_prophet_uncertainty(fcst)

    with sentry_sdk.start_span(
        op="data.anomaly.scores", description="Generate anomaly scores using forecast"
    ) as span:
        fcst = detector.scale_scores(fcst)

    with sentry_sdk.start_span(
        op="data.format", description="Format data for frontend"
    ) as span:
        output = process_output(fcst, granularity)

    return output


@app.route("/health/live", methods=["GET"])
def health_check():
    return "", 200


@app.route("/health/ready", methods=["GET"])
def ready_check():
    if not model_initialized:
        return "Model not initialized", 503
    return "", 200


def aggregate_anomalies(data, granularity):
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
            list(map(int, ts["ds"])), [[{"count": round(x, 5)}] for x in list(ts[value_col])]
        )
        start = int(ts["ds"].iloc[0])
        end = int(ts["ds"].iloc[-1])
        return {"data": list(data), "start": start, "end": end}

    data["ds"] = data["ds"].astype(np.int64) * 1e-9
    anomalies_data = data[~data["anomalies"].isna()][["ds", "anomalies", "y", "yhat"]]
    anomalies = []
    if len(anomalies_data) > 0:
        anomalies = aggregate_anomalies(anomalies_data, granularity)

    results = {
        "y": convert_ts(data, "y"),
        "yhat_upper": convert_ts(data, "yhat_upper"),
        "yhat_lower": convert_ts(data, "yhat_lower"),
        "anomalies": anomalies,
    }
    return results
