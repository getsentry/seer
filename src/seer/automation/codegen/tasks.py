from seer.automation.codegen.models import (
    CodegenContinuation,
    CodegenPrReviewRequest,
    CodegenStatus,
    CodegenUnitTestsRequest,
    CodegenUnitTestsResponse,
    CodegenUnitTestsStateRequest,
)
from seer.automation.codegen.state import CodegenContinuationState
from seer.automation.codegen.unittest_step import UnittestStep, UnittestStepRequest
from seer.automation.state import DbState, DbStateRunTypes
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


def create_initial_unittest_run(request: CodegenUnitTestsRequest) -> DbState[CodegenContinuation]:
    state = CodegenContinuationState.new(
        CodegenContinuation(request=request), group_id=request.pr_id, t=DbStateRunTypes.UNIT_TEST
    )

    with state.update() as cur:
        cur.status = CodegenStatus.PENDING
        cur.signals = []
        cur.mark_triggered()

    return state


@inject
def codegen_unittest(request: CodegenUnitTestsRequest, app_config: AppConfig = injected):
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


def get_unittest_state(request: CodegenUnitTestsStateRequest):
    state = CodegenContinuationState(request.run_id)
    return state.get()


def codegen_pr_review(request: CodegenPrReviewRequest):
    raise NotImplementedError("PR Review is not implemented yet.")
