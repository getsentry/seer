from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixContinuation, AutofixRequest
from seer.automation.autofix.state import ContinuationState
from seer.automation.state import DbState, DbStateRunTypes


def create_initial_autofix_run(request: AutofixRequest) -> DbState[AutofixContinuation]:
    state = ContinuationState.new(
        AutofixContinuation(request=request),
        group_id=request.issue.id,
        t=DbStateRunTypes.AUTOFIX,
    )

    with state.update() as cur:
        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    event_manager.send_root_cause_analysis_will_start()

    return state
