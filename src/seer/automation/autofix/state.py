import dataclasses

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.state import DbState, DbStateRunTypes
from seer.db import DbRunState


@dataclasses.dataclass
class ContinuationState(DbState[AutofixContinuation]):
    id: int
    model: type[AutofixContinuation] = AutofixContinuation
    type: DbStateRunTypes = dataclasses.field(default=DbStateRunTypes.AUTOFIX)

    def apply_to_run_state(self, state: AutofixContinuation, run_state: DbRunState):
        state.mark_updated()
        run_state.updated_at = state.updated_at
        run_state.last_triggered_at = state.last_triggered_at
