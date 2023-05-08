import sentry_sdk
import os

from flask import Flask, request, Response
from sentry_sdk.integrations.flask import FlaskIntegration
import pandas as pd
import numpy as np
import scipy

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


@app.route("/trends/breakpoint-detector", methods=["POST"])
def breakpoint_trends_endpoint(pval=0.01, trend_perc=0.05):

    data = request.get_json()
    txns_data = data['data']

    sort_function = data['sort']
    trend_percentage_list = []

    #defined outside of for loop so error won't throw for empty data 
    transaction_list = txns_data.keys()

    for txn in transaction_list:

        keys = list(txns_data[txn].keys())
        count_data = txns_data[txn]['count()']['data']

        if keys[0] == 'count()':
            ts_data = txns_data[txn][keys[1]]['data']
        else:
            ts_data = txns_data[txn][keys[0]]['data']

        timestamps_zero_filled = [x[0] for x in ts_data]

        #data without zero-filling
        timestamps = []
        metrics = []
        counts = []

        #create lists for time/metric lists without 0 values for more accurate breakpoint analysis
        for i in range(len(ts_data)):
            count = count_data[i][1][0]['count']

            if count != 0:
                counts.append(count)
                timestamps.append(ts_data[i][0])
                metrics.append(ts_data[i][1][0]['count'])

        #snuba query limit was hit and we won't have complete data for this transaction so disregard this txn
        if None in metrics:
            continue

        timeseries = pd.DataFrame(
            {
                'time': timestamps,
                'y': metrics
            }
        )

        #don't include transaction if there are less than two datapoints
        if len(metrics) < 2:
            continue

        change_points = CUSUMDetector(timeseries).detector()

        #sort change points by start time to get most recent one
        change_points.sort(key=lambda x: x.start_time)
        num_breakpoints = len(change_points)

        #if breakpoints are detected, get most recent changepoint
        if num_breakpoints != 0:
            change_point = change_points[-1].start_time
            change_index = timestamps.index(change_point)

        #if breakpoint is in the very beginning or no breakpoints are detected, use midpoint analysis instead
        elif num_breakpoints == 0 or change_index <= 5 or change_index == len(timestamps)-2:
            change_index = int(len(timestamps_zero_filled) / 2)
            change_point = timestamps_zero_filled[change_index]


        first_half = [metrics[i] for i in range(len(metrics)) if timestamps[i] < change_point]
        second_half = [metrics[i] for i in range(len(metrics)) if timestamps[i] >= change_point]

        #if either of the halves don't have any data to compare to then move on to the next txn
        if len(first_half) == 0 or len(second_half) == 0:
            continue

        #get the non-zero counts for the first and second halves
        counts_first_half = [counts[i] for i in range(len(counts)) if timestamps[i] < change_point]
        counts_second_half = [counts[i] for i in range(len(counts)) if timestamps[i] >= change_point]

        mu0 = np.average(first_half)
        mu1 = np.average(second_half)

        #sum of counts before/after changepoint
        count_range_1 = sum(counts_first_half)
        count_range_2 = sum(counts_second_half)

        # calculate t-value between both groups
        scipy_t_test = scipy.stats.ttest_ind(first_half, second_half, equal_var=False)

        if mu0 == 0:
            trend_percentage = mu1
        else:
            trend_percentage = mu1/mu0


        txn_names = txn.split(",")
        output_dict = {
            "project": txn_names[0],
            "transaction": txn_names[1],
            "aggregate_range_1": mu0,
            "aggregate_range_2": mu1,
            "count_range_1": count_range_1,
            "count_range_2": count_range_2,
            "unweighted_t_value": scipy_t_test.statistic,
            "unweighted_p_value": scipy_t_test.pvalue,
            "trend_percentage": trend_percentage,
            "trend_difference": mu1 - mu0,
            "count_percentage": count_range_2/count_range_1,
			"breakpoint": change_point
        }

        #TREND LOGIC:
        #  1. p-value of t-test is less than passed in threshold (default = 0.01)
        #  2. trend percentage is greater than passed in threshold (default = 5%)

        # most improved - get only negatively significant trending txns
        if sort_function == 'trend_percentage()' and mu1 <= mu0 and scipy_t_test.pvalue < pval and abs(trend_percentage-1) > trend_perc:
            trend_percentage_list.append((trend_percentage, output_dict))

        #if most regressed - get only positively signiificant txns
        elif sort_function == '-trend_percentage()' and mu0 <= mu1 and scipy_t_test.pvalue < pval and abs(trend_percentage-1) > trend_perc:
            trend_percentage_list.append((trend_percentage, output_dict))

    if sort_function == 'trend_percentage()':
        sorted_trends = (sorted(trend_percentage_list, key=lambda x: x[0]))
    else:
        sorted_trends = (sorted(trend_percentage_list, key=lambda x: x[0], reverse=True))

    top_trends = {'data': [x[1] for x in sorted_trends]}

    return top_trends


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
