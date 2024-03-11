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
    # Ensure string fields are within their allowed values and apply defaults if missing
    data.sort = data.get("sort", "")
    if data.sort not in ["", "trend_percentage()", "-trend_percentage()"]:
        data.sort = ""

    data.allow_midpoint = data.get("allow_midpoint", "1")
    if data.allow_midpoint not in ["0", "1"]:
        data.allow_midpoint = "1"

    # Ensure validate_tail_hours is a non-negative integer, default to 0 if not provided or invalid
    try:
        data.validate_tail_hours = int(data.get("validate_tail_hours", 0))
    except ValueError:
        data.validate_tail_hours = 0

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
