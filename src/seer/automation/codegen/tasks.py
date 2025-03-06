from seer.automation.codegen.models import (
    CodegenBaseRequest,
    CodegenContinuation,
    CodegenPrClosedRequest,
    CodegenPrClosedResponse,
    CodegenPrReviewResponse,
    CodegenRelevantWarningsRequest,
    CodegenRelevantWarningsResponse,
    CodegenStatus,
    CodegenUnitTestsResponse,
    CodegenUnitTestsStateRequest,
)
from seer.automation.codegen.pr_closed_step import PrClosedStep, PrClosedStepRequest
from seer.automation.codegen.pr_review_step import PrReviewStep, PrReviewStepRequest
from seer.automation.codegen.relevant_warnings_step import (
    RelevantWarningsStep,
    RelevantWarningsStepRequest,
)
from seer.automation.codegen.state import CodegenContinuationState
from seer.automation.codegen.unittest_step import UnittestStep, UnittestStepRequest
from seer.automation.state import DbState, DbStateRunTypes
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


def create_initial_unittest_run(request: CodegenBaseRequest) -> DbState[CodegenContinuation]:
    state = CodegenContinuationState.new(
        CodegenContinuation(request=request), group_id=request.pr_id, t=DbStateRunTypes.UNIT_TEST
    )

    with state.update() as cur:
        cur.status = CodegenStatus.PENDING
        cur.signals = []
        cur.mark_triggered()

    return state


def create_initial_pr_review_run(request: CodegenBaseRequest) -> DbState[CodegenContinuation]:
    state = CodegenContinuationState.new(
        CodegenContinuation(request=request), group_id=request.pr_id, t=DbStateRunTypes.PR_REVIEW
    )

    with state.update() as cur:
        cur.status = CodegenStatus.PENDING
        cur.signals = []
        cur.mark_triggered()

    return state


def create_initial_pr_closed_run(request: CodegenPrClosedRequest) -> DbState[CodegenContinuation]:
    state = CodegenContinuationState.new(
        CodegenContinuation(request=request), group_id=request.pr_id, t=DbStateRunTypes.PR_CLOSED
    )

    with state.update() as cur:
        cur.status = CodegenStatus.PENDING
        cur.signals = []
        cur.mark_triggered()

    return state


def create_initial_relevant_warnings_run(
    request: CodegenRelevantWarningsRequest,
) -> DbState[CodegenContinuation]:
    state = CodegenContinuationState.new(
        CodegenContinuation(request=request),
        group_id=request.pr_id,
        t=DbStateRunTypes.RELEVANT_WARNINGS,
    )

    with state.update() as cur:
        cur.status = CodegenStatus.PENDING
        cur.signals = []
        cur.mark_triggered()

    return state


@inject
def codegen_unittest(request: CodegenBaseRequest, app_config: AppConfig = injected):
    state = create_initial_unittest_run(request)

    cur_state = state.get()
    # Process has no further work.
    # if cur_state.status in CodegenStatus.terminal():
    #     logger.warning(f"Ignoring job, state {cur_state.status}")
    #     return

    unittest_request = UnittestStepRequest(
        run_id=cur_state.run_id,
        pr_id=request.pr_id,
        repo_definition=request.repo,
    )
    UnittestStep.get_signature(unittest_request, queue=app_config.CELERY_WORKER_QUEUE).apply_async()

    return CodegenUnitTestsResponse(run_id=cur_state.run_id)


@inject
def codegen_pr_closed(request: CodegenBaseRequest, app_config: AppConfig = injected):
    state = create_initial_pr_closed_run(request)

    cur_state = state.get()

    pr_closed_request = PrClosedStepRequest(
        run_id=cur_state.run_id,
        pr_id=request.pr_id,
        repo_definition=request.repo,
    )

    PrClosedStep.get_signature(
        pr_closed_request, queue=app_config.CELERY_WORKER_QUEUE
    ).apply_async()

    return CodegenPrClosedResponse(run_id=cur_state.run_id)


def get_unittest_state(request: CodegenUnitTestsStateRequest):
    state = CodegenContinuationState(request.run_id)
    return state.get()


@inject
def codegen_pr_review(request: CodegenBaseRequest, app_config: AppConfig = injected):
    state = create_initial_pr_review_run(request)

    cur_state = state.get()

    pr_review_request = PrReviewStepRequest(
        run_id=cur_state.run_id,
        pr_id=request.pr_id,
        repo_definition=request.repo,
    )

    PrReviewStep.get_signature(
        pr_review_request, queue=app_config.CELERY_WORKER_QUEUE
    ).apply_async()

    return CodegenPrReviewResponse(run_id=cur_state.run_id)


@inject
def codegen_relevant_warnings(
    request: CodegenRelevantWarningsRequest, app_config: AppConfig = injected
):
    state = create_initial_relevant_warnings_run(request)

    cur_state = state.get()

    relevant_warnings_request = RelevantWarningsStepRequest(
        repo=request.repo,
        pr_id=request.pr_id,
        organization_id=request.organization_id,
        warnings=request.warnings,
        commit_sha=request.commit_sha,
        run_id=cur_state.run_id,
        should_post_to_overwatch=True,
    )

    RelevantWarningsStep.get_signature(
        relevant_warnings_request, queue=app_config.CELERY_WORKER_QUEUE
    ).apply_async()

    return CodegenRelevantWarningsResponse(run_id=cur_state.run_id)
