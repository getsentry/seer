import datetime

from johen import generate

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.autofix.state import ContinuationState
from seer.automation.state import DbStateRunTypes


def test_before_update_marks_updated():
    """Test that before_update correctly marks the state as updated with current time"""
    continuation = next(generate(AutofixContinuation))
    initial_time = datetime.datetime(2023, 1, 1)
    continuation.updated_at = initial_time
    state = ContinuationState.new(group_id=1, value=continuation, t=DbStateRunTypes.AUTOFIX)

    with state.update() as cur:
        # Make some change to trigger update
        cur.last_triggered_at = datetime.datetime(2023, 1, 2)

    # After update, updated_at should be more recent than initial_time
    result = state.get()
    assert result.updated_at > initial_time
    assert isinstance(result.updated_at, datetime.datetime)


def test_before_update_always_updates_timestamp():
    """Test that before_update always updates the timestamp, even without changes"""
    continuation = next(generate(AutofixContinuation))
    initial_time = datetime.datetime(2023, 1, 1)
    continuation.updated_at = initial_time
    state = ContinuationState.new(group_id=1, value=continuation, t=DbStateRunTypes.AUTOFIX)

    with state.update():
        # Don't make any changes
        pass

    # Even without changes, updated_at should still be updated
    result = state.get()
    assert result.updated_at > initial_time
    assert isinstance(result.updated_at, datetime.datetime)
