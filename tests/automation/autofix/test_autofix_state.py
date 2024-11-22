import datetime

from johen import generate

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.autofix.state import ContinuationState
from seer.automation.state import DbStateRunTypes


class TestContinuationState:
    def test_update(self):
        continuation = next(generate(AutofixContinuation))
        continuation.updated_at = "2023-01-01"
        continuation.last_triggered_at = "2023-01-02"
        state = ContinuationState.new(group_id=1, value=continuation, t=DbStateRunTypes.AUTOFIX)

        with state.update() as cur:
            cur.updated_at = "2023-01-01"
            cur.last_triggered_at = "2023-01-02"

        assert state.get().updated_at == datetime.datetime(2023, 1, 1, 0, 0)
        assert state.get().last_triggered_at == datetime.datetime(2023, 1, 2, 0, 0)
