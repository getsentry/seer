import dataclasses

from seer.automation.codegen.models import CodegenContinuation
from seer.automation.state import DbState, DbStateRunTypes
from seer.db import DbRunState


@dataclasses.dataclass
class CodegenContinuationState(DbState[CodegenContinuation]):
    id: int
    model: type[CodegenContinuation] = CodegenContinuation
    type: DbStateRunTypes = dataclasses.field(default=DbStateRunTypes.UNIT_TEST)

    def before_update(self, state: CodegenContinuation):
        state.mark_updated()

    def apply_to_run_state(self, state: CodegenContinuation, run_state: DbRunState):
        run_state.updated_at = state.updated_at
        run_state.last_triggered_at = state.last_triggered_at
