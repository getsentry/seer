from celery_app.config import CeleryQueues
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


def create_initial_unittest_run(request: CodegenUnitTestsRequest) -> DbState[CodegenContinuation]:
    state = CodegenContinuationState.new(
        CodegenContinuation(request=request), group_id=request.pr_id, type=DbStateRunTypes.UNIT_TEST
    )

    with state.update() as cur:
        cur.status = CodegenStatus.PENDING
        cur.signals = []
        cur.mark_triggered()

    return state


def codegen_unittest(request: CodegenUnitTestsRequest):
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
    UnittestStep.get_signature(unittest_request, queue=CeleryQueues.DEFAULT).apply_async()

    return CodegenUnitTestsResponse(run_id=cur_state.run_id)


def get_unittest_state(request: CodegenUnitTestsStateRequest):
    state = CodegenContinuationState.from_id(request.run_id, model=CodegenContinuation)
    return state.get()


def codegen_pr_review(request: CodegenPrReviewRequest):
    raise NotImplementedError("PR Review is not implemented yet.")
