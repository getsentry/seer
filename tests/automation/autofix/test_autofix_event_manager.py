from unittest.mock import MagicMock, patch

import pytest
from johen import generate

from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    DefaultStep,
    ProgressType,
    RootCauseStep,
)
from seer.automation.state import LocalMemoryState
from seer.automation.utils import make_kill_signal


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

    @patch("time.sleep")
    def test_reset_steps_to_point(self, mock_time, event_manager, state):
        with state.update() as cur:
            cur.steps = [
                RootCauseStep(id="1", title="Step 1", status=AutofixStatus.COMPLETED),
                DefaultStep(
                    id="2",
                    title="Step 2",
                    status=AutofixStatus.COMPLETED,
                    insights=[
                        MagicMock(spec=InsightSharingOutput, id="insight1"),
                        MagicMock(spec=InsightSharingOutput, id="insight2"),
                        MagicMock(spec=InsightSharingOutput, id="insight3"),
                    ],
                ),
                RootCauseStep(id="3", title="Step 3", status=AutofixStatus.PROCESSING),
                DefaultStep(id="4", title="Step 4", status=AutofixStatus.PROCESSING),
            ]

        # Test case 1: Reset to second step, before any insights
        with patch.object(state, "get", wraps=state.get) as mock_get:
            mock_get.side_effect = [
                state.get(),  # First call returns state with kill signal
                MagicMock(signals=[]),  # Second call returns state without kill signal
            ]
            result = event_manager.reset_steps_to_point(1, None)

        assert result is True
        assert len(state.get().steps) == 2
        assert state.get().steps[-1].id == "2"
        assert len(state.get().steps[-1].insights) == 0

        # Test case 2: Reset to second step, keep only first insight
        with state.update() as cur:
            cur.steps = [
                RootCauseStep(id="1", title="Step 1", status=AutofixStatus.COMPLETED),
                DefaultStep(
                    id="2",
                    title="Step 2",
                    status=AutofixStatus.COMPLETED,
                    insights=[
                        MagicMock(spec=InsightSharingOutput, id="insight1"),
                        MagicMock(spec=InsightSharingOutput, id="insight2"),
                        MagicMock(spec=InsightSharingOutput, id="insight3"),
                    ],
                ),
                RootCauseStep(id="3", title="Step 3", status=AutofixStatus.PROCESSING),
            ]

        with patch.object(state, "get", wraps=state.get) as mock_get:
            mock_get.side_effect = [
                state.get(),  # First call returns state with kill signal
                MagicMock(signals=[]),  # Second call returns state without kill signal
            ]
            result = event_manager.reset_steps_to_point(1, 0)

        assert result is True
        assert len(state.get().steps) == 2
        assert state.get().steps[-1].id == "2"
        assert len(state.get().steps[-1].insights) == 1
        assert state.get().steps[-1].insights[0].id == "insight1"

        # Test case 3: Reset to first step (RootCauseStep)
        with state.update() as cur:
            cur.steps = [
                RootCauseStep(id="1", title="Step 1", status=AutofixStatus.COMPLETED),
                DefaultStep(id="2", title="Step 2", status=AutofixStatus.PROCESSING),
            ]

        with patch.object(state, "get", wraps=state.get) as mock_get:
            mock_get.side_effect = [
                state.get(),  # First call returns state with kill signal
                MagicMock(signals=[]),  # Second call returns state without kill signal
            ]
            result = event_manager.reset_steps_to_point(0, None)

        assert result is True
        assert len(state.get().steps) == 1
        assert state.get().steps[-1].id == "1"

        # Test case 4: Timeout while waiting for steps to be killed
        with patch.object(state, "get", wraps=state.get) as mock_get:
            mock_get.return_value = MagicMock(signals=[make_kill_signal()])
            mock_time.side_effect = [None] * 6  # Simulate timeout
            result = event_manager.reset_steps_to_point(0, None)

        assert result is False
        mock_time.assert_called_with(0.5)
        assert mock_time.call_count == 6

        # Test case 5: Successfully kill steps before timeout
        with patch.object(state, "get", wraps=state.get) as mock_get:
            mock_get.side_effect = [
                MagicMock(signals=[make_kill_signal()]),
                MagicMock(signals=[make_kill_signal()]),
                MagicMock(signals=[make_kill_signal()]),
                MagicMock(signals=[]),
            ]
            mock_time.reset_mock()
            mock_time.side_effect = [None, None, None]  # Simulate successful kill after 3 attempts
            result = event_manager.reset_steps_to_point(0, None)

        assert result is True
        mock_time.assert_called_with(0.5)
        assert mock_time.call_count == 2  # Called twice before signals are cleared
