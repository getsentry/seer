import dataclasses

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.state import DbState, DbStateRunTypes
from seer.db import DbRunState


@dataclasses.dataclass
class ContinuationState(DbState[AutofixContinuation]):
    id: int
    model: type[AutofixContinuation] = AutofixContinuation
    type: DbStateRunTypes = dataclasses.field(default=DbStateRunTypes.AUTOFIX)

    def before_update(self, state: AutofixContinuation):
        state.mark_updated()

    def apply_to_run_state(self, state: AutofixContinuation, run_state: DbRunState):
        """
        This function updates AutofixContinuation state whihc is turn updates the updated_at and
        last_triggered_at fields in the run_state table, i.e. the postgres table for autofix runs.
        """
        run_state.updated_at = state.updated_at
        run_state.last_triggered_at = state.last_triggered_at
