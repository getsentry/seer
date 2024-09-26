import pytest
from johen import generate

from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    ProgressType,
    RootCauseStep,
)
from seer.automation.state import LocalMemoryState


class TestAutofixEventManager:
    @pytest.fixture
    def state(self):
        return LocalMemoryState(AutofixContinuation(request=next(generate(AutofixRequest))))

    @pytest.fixture
    def event_manager(self, state):
        return AutofixEventManager(state)

    def test_on_error_marks_running_steps_errored(self, event_manager, state):
        state.get().steps = [
            RootCauseStep(id="1", title="Step 1", status=AutofixStatus.PROCESSING),
            RootCauseStep(id="2", title="Step 2", status=AutofixStatus.COMPLETED),
            RootCauseStep(id="3", title="Step 3", status=AutofixStatus.PROCESSING),
        ]

        event_manager.on_error("Test error message")

        assert state.get().steps[0].status == AutofixStatus.ERROR
        assert state.get().steps[1].status == AutofixStatus.COMPLETED
        assert state.get().steps[2].status == AutofixStatus.ERROR

    def test_on_error_sets_last_step_completed_message(self, event_manager, state):
        state.get().steps = [
            RootCauseStep(id="1", title="Test Step", status=AutofixStatus.PROCESSING),
        ]

        event_manager.on_error("Test error message")

        assert state.get().steps[-1].completedMessage == "Test error message"

    def test_on_error_sets_state_status_to_error(self, event_manager, state):
        event_manager.on_error("Test error message")

        assert state.get().status == AutofixStatus.ERROR

    def test_on_error_does_not_set_state_status_when_flag_is_false(self, event_manager, state):
        initial_status = state.get().status
        event_manager.on_error("Test error message", should_completely_error=False)

        assert state.get().status == initial_status

    def test_add_log_to_current_step(self, event_manager, state):
        state.get().steps = [
            RootCauseStep(id="1", title="Test Step", status=AutofixStatus.PROCESSING, progress=[])
        ]

        event_manager.add_log("Test log message")

        assert len(state.get().steps[0].progress) == 1
        assert state.get().steps[0].progress[0].message == "Test log message"
        assert state.get().steps[0].progress[0].type == ProgressType.INFO

    def test_add_log_no_processing_step(self, event_manager, state):
        state.get().steps = [
            RootCauseStep(id="1", title="Test Step", status=AutofixStatus.COMPLETED, progress=[])
        ]

        event_manager.add_log("Test log message")

        assert len(state.get().steps[0].progress) == 1

    def test_add_log_empty_steps(self, event_manager, state):
        state.get().steps = []

        event_manager.add_log("Test log message")

        assert state.get().steps == []

    def test_restart_step(self, event_manager, state):
        state.get().steps = [
            RootCauseStep(id="1", title="Step 1", status=AutofixStatus.ERROR),
            RootCauseStep(id="2", title="Step 2", status=AutofixStatus.COMPLETED),
            RootCauseStep(id="3", title="Step 3", status=AutofixStatus.ERROR),
        ]

        event_manager.restart_step(state.get().steps[2])

        assert state.get().steps[0].status == AutofixStatus.ERROR
        assert state.get().steps[1].status == AutofixStatus.COMPLETED
        assert state.get().steps[2].status == AutofixStatus.PROCESSING
        assert state.get().status == AutofixStatus.PROCESSING
