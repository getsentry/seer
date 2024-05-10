import time

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

from celery_app.config import CeleryQueues
from seer.automation.autofix.models import (
    AutofixEndpointResponse,
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
    run_autofix_execution,
    run_autofix_root_cause,
)
from seer.automation.codebase.models import (
    CodebaseStatusCheckRequest,
    CodebaseStatusCheckResponse,
    CreateCodebaseRequest,
    IndexNamespaceTaskRequest,
    RepoAccessCheckRequest,
    RepoAccessCheckResponse,
)
from seer.automation.codebase.tasks import (
    check_repo_access,
    create_codebase_index,
    get_codebase_index_status,
    index_namespace,
)
from seer.bootup import bootup
from seer.grouping.grouping import GroupingRequest, SimilarityBenchmarkResponse, SimilarityResponse
from seer.inference_models import embeddings_model, grouping_lookup
from seer.json_api import json_api, register_json_api_views
from seer.severity.severity_inference import SeverityRequest, SeverityResponse
from seer.trend_detection.trend_detector import BreakpointRequest, BreakpointResponse, find_trends

app = bootup(
    __name__,
    [FlaskIntegration()],
    init_migrations=True,
    async_load_models=True,
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
def similarity_endpoint(data: GroupingRequest) -> SimilarityResponse:
    with sentry_sdk.start_span(op="seer.grouping", description="grouping lookup") as span:
        similar_issues = grouping_lookup().get_nearest_neighbors(data)
    return similar_issues


@json_api("/v0/issues/similarity-embedding-benchmark")
def similarity_embedding_benchmark_endpoint(data: GroupingRequest) -> SimilarityBenchmarkResponse:
    start_time = time.time()
    embedding = grouping_lookup().encode_text(data.stacktrace).astype("float32")
    embedding_list = embedding.tolist()
    end_time = time.time()
    app.logger.debug(f"Embedding generation time: {end_time - start_time} seconds")

    return SimilarityBenchmarkResponse(embedding=embedding_list)


@json_api("/v1/automation/codebase/index/create")
def create_codebase_index_endpoint(data: CreateCodebaseRequest) -> AutofixEndpointResponse:
    namespace_id = create_codebase_index(data.organization_id, data.project_id, data.repo)

    index_namespace.apply_async(
        (
            IndexNamespaceTaskRequest(
                namespace_id=namespace_id,
            ).model_dump(mode="json"),
        ),
        queue=CeleryQueues.CUDA,
    )

    return AutofixEndpointResponse(started=True)


@json_api("/v1/automation/codebase/repo/check-access")
def repo_access_check_endpoint(data: RepoAccessCheckRequest) -> RepoAccessCheckResponse:
    return RepoAccessCheckResponse(has_access=check_repo_access(data.repo))


@json_api("/v1/automation/codebase/index/status")
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


@json_api("/v1/automation/autofix/start")
def autofix_start_endpoint(data: AutofixRequest) -> AutofixEndpointResponse:
    run_autofix_root_cause(data)
    return AutofixEndpointResponse(started=True)


@json_api("/v1/automation/autofix/update")
def autofix_update_endpoint(
    data: AutofixUpdateRequest,
) -> AutofixEndpointResponse:
    if data.payload.type == AutofixUpdateType.SELECT_ROOT_CAUSE:
        run_autofix_execution(data)
    elif data.payload.type == AutofixUpdateType.CREATE_PR:
        run_autofix_create_pr(data)
    return AutofixEndpointResponse(started=True)


@json_api("/v1/automation/autofix/state")
def get_autofix_state_endpoint(data: AutofixStateRequest) -> AutofixStateResponse:
    state = get_autofix_state(data.group_id)

    if state:
        check_and_mark_if_timed_out(state)

    return AutofixStateResponse(
        group_id=data.group_id, state=state.get().model_dump(mode="json") if state else None
    )


@json_api("/v1/automation/autofix/get-state-from-pr")
def get_autofix_state_from_pr_endpoint(data: AutofixPrIdRequest) -> AutofixStateResponse:
    state = get_autofix_state_from_pr_id(data.provider, data.pr_id)

    if state:
        cur = state.get()
        return AutofixStateResponse(
            group_id=cur.request.issue.id, state=cur.model_dump(mode="json")
        )
    return AutofixStateResponse(group_id=None, state=None)


@app.route("/health/live", methods=["GET"])
def health_check():
    from seer.inference_models import models_loading_status

    if models_loading_status() == "failed":
        return "Models failed to load", 500
    return "", 200


@app.route("/health/ready", methods=["GET"])
def ready_check():
    from seer.inference_models import models_loading_status

    status = models_loading_status()
    if status == "failed":
        return "", 500
    if status == "done":
        return "", 200
    return "", 503


register_json_api_views(app)
