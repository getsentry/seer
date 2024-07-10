from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixContinuation, AutofixRequest
from seer.automation.autofix.state import ContinuationState


def create_initial_autofix_run(request: AutofixRequest):
    state = ContinuationState.new(
        AutofixContinuation(request=request),
        group_id=request.issue.id,
    )

    with state.update() as cur:
        cur.mark_triggered()
    cur = state.get()

    event_manager = AutofixEventManager(state)
    event_manager.send_root_cause_analysis_start()

    return state
