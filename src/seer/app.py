import logging
import time

import flask
import sentry_sdk
from flask import Blueprint, Flask, jsonify
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from celery_app.config import CeleryQueues
from seer.anomaly_detection.anomaly_detection import (
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
)
from seer.automation.autofix.models import (
    AutofixEndpointResponse,
    AutofixEvaluationRequest,
    AutofixPrIdRequest,
    AutofixRequest,
    AutofixStateRequest,
    AutofixStateResponse,
    AutofixUpdateRequest,
    AutofixUpdateType,
)
from seer.automation.autofix.tasks import (
    check_and_mark_if_timed_out,
    get_autofix_state,
    get_autofix_state_from_pr_id,
    run_autofix_create_pr,
    run_autofix_evaluation,
    run_autofix_execution,
    run_autofix_root_cause,
)
from seer.automation.codebase.models import (
    CodebaseIndexEndpointResponse,
    CodebaseStatusCheckRequest,
    CodebaseStatusCheckResponse,
    CreateCodebaseRequest,
    IndexNamespaceTaskRequest,
    RepoAccessCheckRequest,
    RepoAccessCheckResponse,
)
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.tasks import (
    create_codebase_index,
    get_codebase_index_status,
    index_namespace,
)
from seer.automation.utils import raise_if_no_genai_consent
from seer.bootup import bootup, module
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected, resolve
from seer.grouping.grouping import (
    BulkCreateGroupingRecordsResponse,
    CreateGroupingRecordsRequest,
    DeleteGroupingRecordsByHashRequest,
    DeleteGroupingRecordsByHashResponse,
    GroupingRequest,
    SimilarityResponse,
)
from seer.inference_models import anomaly_detection, embeddings_model, grouping_lookup
from seer.json_api import json_api
from seer.severity.severity_inference import SeverityRequest, SeverityResponse
from seer.trend_detection.trend_detector import BreakpointRequest, BreakpointResponse, find_trends

app = flask.current_app
blueprint = Blueprint("app", __name__)


@json_api(blueprint, "/v0/issues/severity-score")
def severity_endpoint(data: SeverityRequest) -> SeverityResponse:
    if data.trigger_error:
        raise Exception("oh no")
    elif data.trigger_timeout:
        time.sleep(0.5)

    response = embeddings_model().severity_score(data)
    sentry_sdk.set_tag("severity", str(response.severity))
    return response


@json_api(blueprint, "/trends/breakpoint-detector")
def breakpoint_trends_endpoint(data: BreakpointRequest) -> BreakpointResponse:
    txns_data = data.data

    sort_function = data.sort
    allow_midpoint = data.allow_midpoint == "1"
    validate_tail_hours = data.validate_tail_hours

    min_pct_change = data.trend_percentage
    min_change = data.min_change

    trend_percentage_list = find_trends(
        txns_data,
        sort_function,
        allow_midpoint,
        min_pct_change,
        min_change,
        validate_tail_hours,
    )

    trends = BreakpointResponse(data=[x[1] for x in trend_percentage_list])

    return trends


@json_api(blueprint, "/v0/issues/similar-issues")
def similarity_endpoint(data: GroupingRequest) -> SimilarityResponse:
    with sentry_sdk.start_span(op="seer.grouping", description="grouping lookup"):
        sentry_sdk.set_tag("read_only", data.read_only)
        sentry_sdk.set_tag("stacktrace_len", len(data.stacktrace))
        sentry_sdk.set_tag("request_hash", data.hash)
        similar_issues = grouping_lookup().get_nearest_neighbors(data)
    return similar_issues


@json_api(blueprint, "/v0/issues/similar-issues/grouping-record")
def similarity_grouping_record_endpoint(
    data: CreateGroupingRecordsRequest,
) -> BulkCreateGroupingRecordsResponse:
    sentry_sdk.set_tag(
        "stacktrace_len_sum", sum([len(stacktrace) for stacktrace in data.stacktrace_list])
    )
    success = grouping_lookup().bulk_create_and_insert_grouping_records(data)
    return success


@blueprint.route(
    "/v0/issues/similar-issues/grouping-record/delete/<int:project_id>", methods=["GET"]
)
def delete_grouping_record_endpoint(project_id: int):
    success = grouping_lookup().delete_grouping_records_for_project(project_id)
    return jsonify(success=success)


@json_api(blueprint, "/v0/issues/similar-issues/grouping-record/delete-by-hash")
def delete_grouping_records_by_hash_endpoint(
    data: DeleteGroupingRecordsByHashRequest,
) -> DeleteGroupingRecordsByHashResponse:
    success = grouping_lookup().delete_grouping_records_by_hash(data)
    return success


@json_api(blueprint, "/v1/automation/codebase/index/create")
def create_codebase_index_endpoint(data: CreateCodebaseRequest) -> CodebaseIndexEndpointResponse:
    raise_if_no_genai_consent(data.organization_id)

    namespace_id = create_codebase_index(data.organization_id, data.project_id, data.repo)

    index_namespace.apply_async(
        (
            IndexNamespaceTaskRequest(
                namespace_id=namespace_id,
            ).model_dump(mode="json"),
        ),
        queue=CeleryQueues.CUDA,
    )

    return CodebaseIndexEndpointResponse(started=True)


@json_api(blueprint, "/v1/automation/codebase/repo/check-access")
def repo_access_check_endpoint(data: RepoAccessCheckRequest) -> RepoAccessCheckResponse:
    return RepoAccessCheckResponse(has_access=RepoClient.check_repo_write_access(data.repo))


@json_api(blueprint, "/v1/automation/codebase/index/status")
def get_codebase_index_status_endpoint(
    data: CodebaseStatusCheckRequest,
) -> CodebaseStatusCheckResponse:
    return CodebaseStatusCheckResponse(
        status=get_codebase_index_status(
            organization_id=data.organization_id,
            project_id=data.project_id,
            repo=data.repo,
        )
    )


@json_api(blueprint, "/v1/automation/autofix/start")
def autofix_start_endpoint(data: AutofixRequest) -> AutofixEndpointResponse:
    raise_if_no_genai_consent(data.organization_id)
    run_id = run_autofix_root_cause(data)
    return AutofixEndpointResponse(started=True, run_id=run_id)


@json_api(blueprint, "/v1/automation/autofix/update")
def autofix_update_endpoint(
    data: AutofixUpdateRequest,
) -> AutofixEndpointResponse:
    if data.payload.type == AutofixUpdateType.SELECT_ROOT_CAUSE:
        run_autofix_execution(data)
    elif data.payload.type == AutofixUpdateType.CREATE_PR:
        run_autofix_create_pr(data)
    return AutofixEndpointResponse(started=True, run_id=data.run_id)


@json_api(blueprint, "/v1/automation/autofix/state")
def get_autofix_state_endpoint(data: AutofixStateRequest) -> AutofixStateResponse:
    state = get_autofix_state(group_id=data.group_id, run_id=data.run_id)

    if state:
        check_and_mark_if_timed_out(state)

        cur_state = state.get()

        return AutofixStateResponse(
            group_id=cur_state.request.issue.id,
            run_id=cur_state.run_id,
            state=cur_state.model_dump(mode="json"),
        )

    return AutofixStateResponse(group_id=None, run_id=None, state=None)


@json_api(blueprint, "/v1/automation/autofix/state/pr")
def get_autofix_state_from_pr_endpoint(data: AutofixPrIdRequest) -> AutofixStateResponse:
    state = get_autofix_state_from_pr_id(data.provider, data.pr_id)

    if state:
        cur_state = state.get()
        return AutofixStateResponse(
            group_id=cur_state.request.issue.id,
            run_id=cur_state.run_id,
            state=cur_state.model_dump(mode="json"),
        )
    return AutofixStateResponse(group_id=None, run_id=None, state=None)


@json_api(blueprint, "/v1/automation/autofix/evaluations/start")
def autofix_evaluation_start_endpoint(data: AutofixEvaluationRequest) -> AutofixEndpointResponse:
    config = resolve(AppConfig)
    if not config.DEV:
        raise RuntimeError("The evaluation endpoint is only available in development mode")

    run_autofix_evaluation(data.dataset_name, data.run_name, is_test=data.test)

    return AutofixEndpointResponse(started=True, run_id=-1)


@json_api(blueprint, "/v1/anomaly-detection/detect")
def detect_anomalies_endpoint(data: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
    return anomaly_detection().detect_anomalies(data)


@json_api(blueprint, "/v1/anomaly-detection/store")
def store_data_endpoint(data: StoreDataRequest) -> StoreDataResponse:
    return anomaly_detection().store_data(data)


@blueprint.route("/health/live", methods=["GET"])
def health_check():
    from seer.inference_models import models_loading_status

    if models_loading_status() == "failed":
        return "Models failed to load", 500
    return "", 200


@blueprint.route("/health/ready", methods=["GET"])
def ready_check():
    from seer.inference_models import models_loading_status

    status = models_loading_status()
    if status == "failed":
        return "", 500
    if status == "done":
        return "", 200
    return "", 503


@module.provider
def base_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    return app


@inject
def start_app(app: Flask = injected) -> Flask:
    bootup(
        start_model_loading=True,
        integrations=[
            FlaskIntegration(),
            LoggingIntegration(
                level=logging.DEBUG,  # Capture debug and above as breadcrumbs
            ),
        ],
    )
    return app
