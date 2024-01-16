import functools
import os
import time
from typing import Any, Callable, List

import sentry_sdk
from flask import Flask
from sentry_sdk.integrations.flask import FlaskIntegration

from seer.grouping.grouping import GroupingLookup, GroupingRequest, GroupingResponse
from seer.json_api import json_api, register_json_api_views
from seer.severity.severity_inference import SeverityInference, SeverityRequest, SeverityResponse
from seer.trend_detection.trend_detector import BreakpointRequest, BreakpointResponse, find_trends


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


@functools.cache
def embeddings_model() -> SeverityInference:
    return SeverityInference(
        model_path("issue_severity_v0/embeddings"), model_path("issue_severity_v0/classifier")
    )


@functools.cache
def grouping_lookup() -> GroupingLookup:
    return GroupingLookup(
        model_path=model_path("issue_grouping_v0/model"),
        data_path=model_path("issue_grouping_v0/data.pkl"),
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
    txns_data = data.data

    sort_function = data.sort
    allow_midpoint = data.allow_midpoint == "1"
    validate_tail_hours = data.validate_tail_hours

    min_pct_change = data.trend_percentage
    min_change = data.min_change

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

    trends = BreakpointResponse(data=[x[1] for x in trend_percentage_list])
    app.logger.debug("Trend results: %s", trends)

    return trends


@json_api("/v0/issues/similar-issues")
def grouping_endpoint(data: GroupingRequest) -> List[GroupingResponse]:
    with sentry_sdk.start_span(op="seer.grouping", description="grouping lookup") as span:
        grouping_results = grouping_lookup().get_nearest_neighbors(data)
    return grouping_results


@app.route("/health/live", methods=["GET"])
def health_check():
    return "", 200


@app.route("/health/ready", methods=["GET"])
def ready_check():
    return "", 200


register_json_api_views(app)


def run(environ: dict, start_response: Callable) -> Any:
    # Force preload
    embeddings_model()
    grouping_lookup()
    return app(environ, start_response)
