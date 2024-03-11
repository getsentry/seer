import datetime
import json
import os
import time

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

from seer.automation.autofix.models import AutofixEndpointResponse, AutofixRequest
from seer.automation.autofix.tasks import run_autofix
from seer.bootup import bootup
from seer.db import ProcessRequest, Session
from seer.grouping.grouping import GroupingRequest, SimilarityResponse
from seer.inference_models import embeddings_model, grouping_lookup
from seer.json_api import json_api, register_json_api_views
from seer.severity.severity_inference import SeverityRequest, SeverityResponse
from seer.trend_detection.trend_detector import BreakpointRequest, BreakpointResponse, find_trends

app = bootup(
    __name__,
    [FlaskIntegration()],
    init_migrations=True,
    eager_load_inference_models=os.environ.get("LAZY_INFERENCE_MODELS") != "1",
)


@json_api("/v0/issues/severity-score")
def severity_endpoint(data: SeverityRequest) -> SeverityResponse:
    if data.trigger_error:
        raise Exception("oh no")
    elif data.trigger_timeout:
        time.sleep(0.5)

    with sentry_sdk.start_span(
        op="seer.severity",
        description="Generate issue severity score",
    ) as span:
        response = embeddings_model().severity_score(data)
        span.set_tag("severity", str(response.severity))
    return response


@json_api("/trends/breakpoint-detector")
def breakpoint_trends_endpoint(data: BreakpointRequest) -> BreakpointResponse:
    """
    Endpoint to detect trends and breakpoints in time series data.

    Parameters:
    - `data`: BreakpointRequest object containing the following fields:
        - `data`: Mapping of transaction names to their respective data points.
        - `sort`: Sort order for trend percentages. Accepted values are "", "trend_percentage()", or "-trend_percentage()". Default is "".
        - `allow_midpoint`: A string indicating whether midpoint analysis is allowed. Accepted values are "0" (no) and "1" (yes). Default is "1".
        - `validate_tail_hours`: An integer specifying the tail hours to validate. Should be non-negative. Default is 0.
        - `trend_percentage`: The minimum trend percentage change to report. Default is 0.1.
        - `min_change`: The minimum trend change to report. Default is 0.0.

    Returns:
    A `BreakpointResponse` object containing detected trends.

    """

@json_api("/v0/issues/similar-issues")
def similarity_endpoint(data: GroupingRequest) -> SimilarityResponse:
    with sentry_sdk.start_span(op="seer.grouping", description="grouping lookup") as span:
        similar_issues = grouping_lookup().get_nearest_neighbors(data)
    return similar_issues


@json_api("/v0/automation/autofix")
def autofix_endpoint(data: AutofixRequest) -> AutofixEndpointResponse:
    run_autofix.delay(data.model_dump(mode="json"))
    return AutofixEndpointResponse(started=True)


@app.route("/health/live", methods=["GET"])
def health_check():
    return "", 200


@app.route("/health/ready", methods=["GET"])
def ready_check():
    return "", 200


register_json_api_views(app)
