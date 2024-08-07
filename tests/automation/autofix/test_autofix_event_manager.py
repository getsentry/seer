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

        assert len(state.get().steps[0].progress) == 0

    def test_add_log_empty_steps(self, event_manager, state):
        state.get().steps = []

        event_manager.add_log("Test log message")

        assert state.get().steps == []
