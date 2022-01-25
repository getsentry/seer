import logging
import time

from flask import Flask, request
import pandas as pd

from prophet_detector import ProphetDetector, ProphetParams

log = logging.getLogger()
app = Flask(__name__)


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
        uncertainty_samples=None,
    )
    m = ProphetDetector(pd.DataFrame(data["train"]), start, end, params)
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
    fcst["ds"] = (fcst["ds"].astype(np.int64) * 1e-9).astype(np.int64)
    timings["gen_scores"] = time.time() - s

    s = time.time()
    output = process_output(fcst)
    timings["format_output"] = time.time() - s

    logging.info(timings)

    return output


def convert_ts(ts, value_col):
    data = zip(
        list(ts["ds"]),
        [[{"count": x}] for x in list(ts[value_col])],
    )
    start = ts["ds"].iloc[0]
    end = ts["ds"].iloc[-1]
    return {"data": list(data), "start": start, "end": end}


def aggregate_anomalies(data):
    anomalies = []
    last_score = None
    anomaly_index = -1
    granularity = 5 * 60  # TODO get from class
    for ds_time, score in data[~data["anomalies"].isna()][["ds", "anomalies"]].itertuples(index=False):
        if score == last_score:
            anomalies[anomaly_index]["end"] = ds_time + granularity
        else:
            anomaly_index += 1
            anomalies.append(
                {
                    "start": ds_time,
                    "end": ds_time + granularity,
                    "confidence": score,
                    "id": anomaly_index,
                }
            )
    last_score = score

    return anomalies


def process_output(data):
    results = {
        "y": convert_ts(data, "y"),
        "yhat_upper": convert_ts(data, "yhat_upper"),
        "yhat_lower": convert_ts(data, "yhat_lower"),
        # TODO: remove (this is just for debugging)
        "anomaly_scores": convert_ts(data, "final_score"),
        "anomalies": aggregate_anomalies(data),
    }
    return results
