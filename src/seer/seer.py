import os
import time

import pandas as pd
import sentry_sdk
from sentry_sdk import start_transaction
from flask import Flask, request
from sentry_sdk.integrations.flask import FlaskIntegration

from seer.trend_detection.trend_detector import find_trends


def traces_sampler(sampling_context):
    if sampling_context["parent_sampled"] is not None:
        return sampling_context["parent_sampled"]

    if "wsgi_environ" in sampling_context:
        path_info = sampling_context["wsgi_environ"].get("PATH_INFO")
        if path_info and path_info.startswith("/health/"):
            return 0.0

    return 1.0


sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    integrations=[FlaskIntegration()],
    traces_sampler=traces_sampler,
    profiles_sample_rate=1.0,
    enable_tracing=True,
)
app = Flask(__name__)
root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))


def model_path(subpath: str) -> str:
    return os.path.join(root, "models", subpath)


model_initialized = False
detector = None
embeddings_model = None
if not os.environ.get("PYTEST_CURRENT_TEST"):
    from seer.anomaly_detection.prophet_detector import ProphetDetector
    from seer.anomaly_detection.prophet_params import ProphetParams
    from seer.severity.severity_inference import SeverityInference

    MODEL_PARAMS = ProphetParams(
        interval_width=0.975,
        changepoint_prior_scale=0.01,
        weekly_seasonality=14,
        daily_seasonality=False,
        uncertainty_samples=None,
    )

    detector = ProphetDetector(MODEL_PARAMS)
    embeddings_model = SeverityInference(
        model_path("issue_severity_v0/embeddings"), model_path("issue_severity_v0/classifier")
    )
    model_initialized = True


@app.route("/v0/issues/severity-score", methods=["POST"])
def severity_endpoint():
    with start_transaction(op="endpoint", name="Severity Score Endpoint"):
        data = request.get_json()
        if data.get("trigger_error") is not None:
            raise Exception("oh no")
        elif data.get("trigger_timeout") is not None:
            time.sleep(0.5)
        severity = embeddings_model.severity_score(data)
        results = {"severity": str(severity)}
        return results


@app.route("/trends/breakpoint-detector", methods=["POST"])
def breakpoint_trends_endpoint():
    data = request.get_json()
    txns_data = data["data"]

    sort_function = data.get("sort", "")
    allow_midpoint = data.get("allow_midpoint", "1") == "1"
    validate_tail_hours = data.get("validate_tail_hours", 0)

    min_pct_change = float(data.get("trend_percentage()", 0.1))
    min_change = float(data.get("min_change()", 0))

    with sentry_sdk.start_span(
        op="cusum.detection",
        description="Get the breakpoint and t-value for every transaction",
    ) as span:
        trend_percentage_list = find_trends(
            txns_data,
            sort_function,
            allow_midpoint,
            min_pct_change,
            min_change,
            validate_tail_hours,
        )

    trends = {"data": [x[1] for x in trend_percentage_list]}
    app.logger.debug("Trend results: %s", trends)

    return trends


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
        op="data.preprocess",
        description="Preprocess data to prepare for anomaly detection",
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

    with sentry_sdk.start_span(op="model.train", description="Train forecasting model") as span:
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

    with sentry_sdk.start_span(op="data.format", description="Format data for frontend") as span:
        output = detector.process_output(fcst, granularity)

    return output


@app.route("/health/live", methods=["GET"])
def health_check():
    return "", 200


@app.route("/health/ready", methods=["GET"])
def ready_check():
    if not model_initialized:
        return "Model not initialized", 503
    return "", 200
