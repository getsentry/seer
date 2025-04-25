import logging
import os
import time

import datadog
import flask
import sentry_sdk
from datadog.dogstatsd.base import statsd
from flask import Blueprint, Flask, jsonify
from openai import APITimeoutError
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from werkzeug.exceptions import GatewayTimeout, InternalServerError

from integrations.codecov.codecov_auth import CodecovAuthentication
from seer.anomaly_detection.models.external import (
    DeleteAlertDataRequest,
    DeleteAlertDataResponse,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    StoreDataRequest,
    StoreDataResponse,
)
from seer.automation.assisted_query.assisted_query import translate_query
from seer.automation.assisted_query.create_cache import create_cache
from seer.automation.assisted_query.models import (
    CreateCacheRequest,
    CreateCacheResponse,
    TranslateRequest,
    TranslateResponse,
)
from seer.automation.autofix.models import (
    AutofixEndpointResponse,
    AutofixEvaluationRequest,
    AutofixPrIdRequest,
    AutofixRequest,
    AutofixStateRequest,
    AutofixStateResponse,
    AutofixUpdateEndpointResponse,
    AutofixUpdateRequest,
    AutofixUpdateType,
)
from seer.automation.autofix.runs import update_repo_access
from seer.automation.autofix.tasks import (
    check_and_mark_if_timed_out,
    comment_on_thread,
    get_autofix_state,
    get_autofix_state_from_pr_id,
    receive_feedback,
    receive_user_message,
    resolve_comment_thread,
    restart_from_point_with_feedback,
    run_autofix_coding,
    run_autofix_evaluation,
    run_autofix_push_changes,
    run_autofix_root_cause,
    run_autofix_solution,
    update_code_change,
)
from seer.automation.codebase.models import RepoAccessCheckRequest, RepoAccessCheckResponse
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.evals.models import CodegenRelevantWarningsEvaluationRequest
from seer.automation.codegen.evals.tasks import run_relevant_warnings_evaluation
from seer.automation.codegen.models import (
    CodecovTaskRequest,
    CodegenBaseRequest,
    CodegenBaseResponse,
    CodegenPrClosedResponse,
    CodegenPrReviewResponse,
    CodegenPrReviewStateRequest,
    CodegenPrReviewStateResponse,
    CodegenRelevantWarningsRequest,
    CodegenRelevantWarningsResponse,
    CodegenUnitTestsResponse,
    CodegenUnitTestsStateRequest,
    CodegenUnitTestsStateResponse,
)
from seer.automation.codegen.tasks import (
    codegen_pr_closed,
    codegen_pr_review,
    codegen_relevant_warnings,
    codegen_retry_unittest,
    codegen_unittest,
    get_unittest_state,
)
from seer.automation.preferences import (
    GetSeerProjectPreferenceRequest,
    GetSeerProjectPreferenceResponse,
    SetSeerProjectPreferenceRequest,
    SetSeerProjectPreferenceResponse,
    get_seer_project_preference,
    set_seer_project_preference,
)
from seer.automation.summarize.issue import run_fixability_score, run_summarize_issue
from seer.automation.summarize.models import (
    GetFixabilityScoreRequest,
    SummarizeIssueRequest,
    SummarizeIssueResponse,
    SummarizeTraceRequest,
    SummarizeTraceResponse,
)
from seer.automation.summarize.traces import summarize_trace
from seer.automation.utils import ConsentError, raise_if_no_genai_consent
from seer.bootup import bootup, module
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected, resolve
from seer.exceptions import ClientError, ServerError
from seer.grouping.grouping import (
    BulkCreateGroupingRecordsResponse,
    CreateGroupingRecordsRequest,
    DeleteGroupingRecordsByHashRequest,
    DeleteGroupingRecordsByHashResponse,
    GroupingRequest,
    SimilarityResponse,
)
from seer.inference_models import (
    autofixability_model,
    embeddings_model,
    grouping_lookup,
    load_anomaly_detection,
    test_grouping_model,
)
from seer.json_api import json_api
from seer.loading import LoadingResult
from seer.severity.severity_inference import SeverityRequest, SeverityResponse
from seer.smoke_test import check_smoke_test
from seer.tags import AnomalyDetectionTags
from seer.trend_detection.trend_detector import BreakpointRequest, BreakpointResponse, find_trends
from seer.workflows.compare.models import CompareCohortsRequest, CompareCohortsResponse
from seer.workflows.compare.service import compare_cohort

logger = logging.getLogger(__name__)

app = flask.current_app
blueprint = Blueprint("app", __name__)
app_module = module

# Initialize Datadog client for metrics
datadog.initialize(
    statsd_host=os.environ.get("STATSD_HOST", "127.0.0.1"),
    statsd_port=int(os.environ.get("STATSD_PORT", "8126")),
)
# Workaround for https://github.com/DataDog/datadogpy/issues/764 as described in https://github.com/getsentry/sentry/pull/68644/files#
statsd.disable_telemetry()
statsd.disable_buffering = False
statsd._container_id = None


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
    with sentry_sdk.start_span(op="seer.grouping", description="grouping lookup") as span:
        sentry_sdk.set_tag("read_only", data.read_only)
        sentry_sdk.set_tag("request_hash", data.hash)
        span.set_data("stacktrace_len", len(data.stacktrace))
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


@json_api(blueprint, "/v1/automation/codebase/repo/check-access")
def repo_access_check_endpoint(data: RepoAccessCheckRequest) -> RepoAccessCheckResponse:
    return RepoAccessCheckResponse(
        has_access=RepoClient.check_repo_write_access(data.repo) or False
    )


@json_api(blueprint, "/v1/automation/autofix/start")
def autofix_start_endpoint(data: AutofixRequest) -> AutofixEndpointResponse:
    raise_if_no_genai_consent(data.organization_id)
    run_id = run_autofix_root_cause(data)
    return AutofixEndpointResponse(started=True, run_id=run_id or -1)


@json_api(blueprint, "/v1/automation/autofix/update")
def autofix_update_endpoint(
    data: AutofixUpdateRequest,
) -> AutofixUpdateEndpointResponse:
    if data.payload.type == AutofixUpdateType.SELECT_ROOT_CAUSE:
        run_autofix_solution(data)
    elif data.payload.type == AutofixUpdateType.SELECT_SOLUTION:
        run_autofix_coding(data)
    elif data.payload.type == AutofixUpdateType.CREATE_PR:
        run_autofix_push_changes(data)
    elif data.payload.type == AutofixUpdateType.CREATE_BRANCH:
        run_autofix_push_changes(data)
    elif data.payload.type == AutofixUpdateType.USER_MESSAGE:
        receive_user_message(data)
    elif data.payload.type == AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK:
        restart_from_point_with_feedback(data)
    elif data.payload.type == AutofixUpdateType.UPDATE_CODE_CHANGE:
        update_code_change(data)
    elif data.payload.type == AutofixUpdateType.COMMENT_THREAD:
        comment_on_thread(data)
    elif data.payload.type == AutofixUpdateType.RESOLVE_COMMENT_THREAD:
        resolve_comment_thread(data)
    elif data.payload.type == AutofixUpdateType.FEEDBACK:
        receive_feedback(data)

    return AutofixUpdateEndpointResponse(run_id=data.run_id)


@json_api(blueprint, "/v1/automation/autofix/state")
def get_autofix_state_endpoint(data: AutofixStateRequest) -> AutofixStateResponse:
    state = get_autofix_state(group_id=data.group_id, run_id=data.run_id)

    if state:
        check_and_mark_if_timed_out(state)

        if data.check_repo_access:
            update_repo_access(state)

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

    run_autofix_evaluation(data)

    return AutofixEndpointResponse(started=True, run_id=-1)


@json_api(blueprint, "/v1/automation/codegen/unit-tests")
def codegen_unit_tests_endpoint(data: CodegenBaseRequest) -> CodegenUnitTestsResponse:
    return codegen_unittest(data)


@json_api(blueprint, "/v1/automation/codegen/pr-closed")
def codegen_pr_closed_endpoint(data: CodegenBaseRequest) -> CodegenPrClosedResponse:
    return codegen_pr_closed(data)


@json_api(blueprint, "/v1/automation/codegen/unit-tests/state")
def codegen_unit_tests_state_endpoint(
    data: CodegenUnitTestsStateRequest,
) -> CodegenUnitTestsStateResponse:
    state = get_unittest_state(data)

    return CodegenUnitTestsStateResponse(
        run_id=state.run_id,
        status=state.status,
        changes=state.file_changes,
        triggered_at=state.last_triggered_at,
        updated_at=state.updated_at,
        completed_at=state.completed_at,
    )


@json_api(blueprint, "/v1/automation/codegen/relevant-warnings")
def codegen_relevant_warnings_endpoint(
    data: CodegenRelevantWarningsRequest,
) -> CodegenRelevantWarningsResponse:
    return codegen_relevant_warnings(data)


@json_api(blueprint, "/v1/automation/codegen/relevant-warnings/evaluation/start")
def codegen_relevant_warnings_evaluation_start_endpoint(
    data: CodegenRelevantWarningsEvaluationRequest,
) -> CodegenRelevantWarningsResponse:
    return run_relevant_warnings_evaluation(data)


@json_api(blueprint, "/v1/automation/codegen/pr-review")
def codegen_pr_review_endpoint(data: CodegenBaseRequest) -> CodegenPrReviewResponse:
    return codegen_pr_review(data)


@json_api(blueprint, "/v1/automation/codegen/pr-review/state")
def codegen_pr_review_state_endpoint(
    data: CodegenPrReviewStateRequest,
) -> CodegenPrReviewStateResponse:
    raise NotImplementedError("PR Review state is not implemented yet.")


@json_api(blueprint, "/v1/automation/codecov-request")
def codecov_request_endpoint(
    data: CodecovTaskRequest,
) -> CodegenBaseResponse:
    is_valid = CodecovAuthentication.authenticate_codecov_app_install(
        data.external_owner_id, data.data.repo.external_id
    )

    if not is_valid:
        raise ConsentError(f"Invalid permissions for org {data.external_owner_id}.")

    if data.request_type == "pr-review":
        return codegen_pr_review_endpoint(data.data)
    elif data.request_type == "unit-tests":
        return codegen_unittest(data.data, is_codecov_request=True)
    elif data.request_type == "pr-closed":
        return codegen_pr_closed_endpoint(data.data)
    elif data.request_type == "retry-unit-tests":
        return codegen_retry_unittest(data.data)

    raise ValueError(f"Unsupported request_type: {data.request_type}")


@json_api(blueprint, "/v1/project-preference")
def get_seer_project_preference_endpoint(
    data: GetSeerProjectPreferenceRequest,
) -> GetSeerProjectPreferenceResponse:
    return get_seer_project_preference(data)


@json_api(blueprint, "/v1/project-preference/set")
def set_seer_project_preference_endpoint(
    data: SetSeerProjectPreferenceRequest,
) -> SetSeerProjectPreferenceResponse:
    return set_seer_project_preference(data)


@json_api(blueprint, "/v1/automation/summarize/issue")
def summarize_issue_endpoint(data: SummarizeIssueRequest) -> SummarizeIssueResponse:
    try:
        return run_summarize_issue(data)
    except APITimeoutError as e:
        raise GatewayTimeout from e
    except Exception as e:
        logger.exception("Error summarizing issue")
        raise InternalServerError from e


@json_api(blueprint, "/v1/automation/summarize/trace")
def summarize_trace_endpoint(data: SummarizeTraceRequest) -> SummarizeTraceResponse:
    try:
        response = summarize_trace(data)
        statsd.increment("seer.automation.summarize.trace.success")
        return response
    except APITimeoutError as e:
        statsd.increment("seer.automation.summarize.trace.api_timeout")
        raise GatewayTimeout from e
    except Exception as e:
        statsd.increment("seer.automation.summarize.trace.server_error")
        logger.exception("Error summarizing trace")
        raise InternalServerError from e


@json_api(blueprint, "/v1/automation/summarize/fixability")
def get_fixability_score_endpoint(data: GetFixabilityScoreRequest) -> SummarizeIssueResponse:
    model = autofixability_model()
    try:
        return run_fixability_score(data, model)
    except APITimeoutError as e:
        raise GatewayTimeout from e
    except Exception as e:
        logger.exception("Error calculating fixability score")
        raise InternalServerError from e


@json_api(blueprint, "/v1/anomaly-detection/detect")
@sentry_sdk.trace
def detect_anomalies_endpoint(data: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
    sentry_sdk.set_tag(AnomalyDetectionTags.SEER_FUNCTIONALITY, "anomaly_detection")
    sentry_sdk.set_tag("organization_id", data.organization_id)
    sentry_sdk.set_tag("project_id", data.project_id)
    try:
        with statsd.timed("seer.anomaly_detection.detect.duration"):
            response = load_anomaly_detection().detect_anomalies(data)
            statsd.increment("seer.anomaly_detection.detect.success")
    except ClientError as e:
        statsd.increment("seer.anomaly_detection.detect.client_error")
        response = DetectAnomaliesResponse(success=False, message=str(e))
    except ServerError:
        statsd.increment("seer.anomaly_detection.detect.server_error")
        raise

    return response


@json_api(blueprint, "/v1/anomaly-detection/store")
@sentry_sdk.trace
def store_data_endpoint(data: StoreDataRequest) -> StoreDataResponse:
    sentry_sdk.set_tag(AnomalyDetectionTags.SEER_FUNCTIONALITY, "anomaly_detection")
    sentry_sdk.set_tag("organization_id", data.organization_id)
    sentry_sdk.set_tag("project_id", data.project_id)
    sentry_sdk.set_tag("alert_id", data.alert.id)
    try:
        with statsd.timed("seer.anomaly_detection.store.duration"):
            response = load_anomaly_detection().store_data(data)
            statsd.increment("seer.anomaly_detection.store.success")
    except ClientError as e:
        statsd.increment("seer.anomaly_detection.store.client_error")
        response = StoreDataResponse(success=False, message=str(e))
    except ServerError:
        statsd.increment("seer.anomaly_detection.store.server_error")
        raise

    return response


@json_api(blueprint, "/v1/anomaly-detection/delete-alert-data")
@sentry_sdk.trace
def delete_alert__data_endpoint(
    data: DeleteAlertDataRequest,
) -> DeleteAlertDataResponse:
    sentry_sdk.set_tag(AnomalyDetectionTags.SEER_FUNCTIONALITY, "anomaly_detection")
    sentry_sdk.set_tag("organization_id", data.organization_id)
    if data.project_id is not None:
        sentry_sdk.set_tag("project_id", data.project_id)
    sentry_sdk.set_tag("alert_id", data.alert.id)
    try:
        with statsd.timed("seer.anomaly_detection.delete_alert_data.duration"):
            response = load_anomaly_detection().delete_alert_data(data)
            statsd.increment("seer.anomaly_detection.delete_alert_data.success")
    except ClientError as e:
        statsd.increment("seer.anomaly_detection.delete_alert_data.client_error")
        response = DeleteAlertDataResponse(success=False, message=str(e))
    except ServerError:
        statsd.increment("seer.anomaly_detection.delete_alert_data.server_error")
        raise

    return response


@json_api(blueprint, "/v1/anomaly-detection/compare-cohorts")
def compare_cohorts_endpoint(
    data: CompareCohortsRequest,
) -> CompareCohortsResponse:
    return compare_cohort(data)


@json_api(blueprint, "/v1/assisted-query/create-cache")
@sentry_sdk.trace
def create_cache_endpoint(data: CreateCacheRequest) -> CreateCacheResponse:
    try:
        with statsd.timed("seer.automation.assisted_query.create_cache.duration"):
            response = create_cache(data)
            statsd.increment("seer.automation.assisted_query.create_cache.success")
    except APITimeoutError as e:
        statsd.increment("seer.automation.assisted_query.create_cache.api_timeout")
        raise GatewayTimeout from e
    except Exception as e:
        statsd.increment("seer.automation.assisted_query.create_cache.server_error")
        logger.exception("Error creating cache")
        raise InternalServerError from e

    return response


@json_api(blueprint, "/v1/assisted-query/translate")
def translate_endpoint(data: TranslateRequest) -> TranslateResponse:
    try:
        with statsd.timed("seer.automation.assisted_query.translate.duration"):
            response = translate_query(data)
            statsd.increment("seer.automation.assisted_query.translate.success")
    except APITimeoutError as e:
        statsd.increment("seer.automation.assisted_query.translate.api_timeout")
        raise GatewayTimeout from e
    except Exception as e:
        statsd.increment("seer.automation.assisted_query.translate.error")
        logger.exception("Error translating query")
        raise InternalServerError from e

    return response


@blueprint.route("/health/live", methods=["GET"])
@inject
def health_check(app_config: AppConfig = injected):
    from seer.inference_models import models_loading_status

    status = models_loading_status()

    if status == LoadingResult.FAILED:
        statsd.increment("seer.health.live.500")
        return "Models failed to load", 500

    # Only run model tests if models are already loaded
    if status == LoadingResult.DONE:
        if app_config.is_grouping_enabled and not test_grouping_model():
            return "Grouping model inference failed", 500

    statsd.increment("seer.health.live.200")
    return "", 200


@blueprint.route("/health/ready", methods=["GET"])
@inject
def ready_check(app_config: AppConfig = injected):
    from seer.inference_models import models_loading_status

    status = models_loading_status()
    if app_config.SMOKE_CHECK:
        smoke_status = check_smoke_test()
        logger.info(f"Celery smoke status: {smoke_status}")
        status = min(status, smoke_status)

    # Only run model tests if models are already loaded
    if status == LoadingResult.DONE:
        if app_config.is_grouping_enabled and not test_grouping_model():
            return "Grouping model inference failed", 500

    if status == LoadingResult.FAILED:
        statsd.increment("seer.health.ready.500")
        return "", 500
    if status == LoadingResult.DONE:
        statsd.increment("seer.health.ready.200")
        return "", 200
    statsd.increment("seer.health.ready.503")
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
