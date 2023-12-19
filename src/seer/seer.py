import os
import time

import pandas as pd
import sentry_sdk
from sentry_sdk import start_transaction
from flask import Flask, request
from sentry_sdk.integrations.flask import FlaskIntegration

from seer.trend_detection.trend_detector import find_trends


def traces_sampler(sampling_context):
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
embeddings_model = None
if not os.environ.get("PYTEST_CURRENT_TEST"):
    from seer.severity.severity_inference import SeverityInference

    embeddings_model = SeverityInference(
        model_path("issue_severity_v0/embeddings"), model_path("issue_severity_v0/classifier")
    )
    model_initialized = True


@app.route("/v0/issues/severity-score", methods=["POST"])
def severity_endpoint():
    data = request.get_json()
    if data.get("trigger_error") is not None:
        raise Exception("oh no")
    elif data.get("trigger_timeout") is not None:
        time.sleep(0.5)
    
    with sentry_sdk.start_span(
        op="seer.severity",
        description="Generate issue severity score",
    ) as span:
        severity = embeddings_model.severity_score(data)
        span.set_tag("severity", str(severity))
        
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
        op="seer.breakpoint_detection",
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


@app.route("/health/live", methods=["GET"])
def health_check():
    return "", 200


@app.route("/health/ready", methods=["GET"])
def ready_check():
    if not model_initialized:
        return "Model not initialized", 503
    return "", 200
