from unittest.mock import MagicMock, patch

import pytest
from johen import generate

from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.components.solution.models import SolutionOutput, SolutionTimelineEvent
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixSolutionUpdatePayload,
    AutofixStatus,
    ChangesStep,
    CodebaseState,
    DefaultStep,
    ProgressType,
    RootCauseStep,
)
from seer.automation.models import FileChange
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
                MagicMock(signals=[]),
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
                MagicMock(signals=[]),
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
                MagicMock(signals=[]),
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
                MagicMock(signals=[]),
            ]
            mock_time.reset_mock()
            mock_time.side_effect = [None, None, None]  # Simulate successful kill after 3 attempts
            result = event_manager.reset_steps_to_point(0, None)

        assert result is True
        mock_time.assert_called_with(0.5)
        assert mock_time.call_count == 2  # Called twice before signals are cleared

    @patch("time.sleep")
    def test_reset_steps_to_point_clears_file_changes(self, mock_time, event_manager, state):
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
                DefaultStep(id="3", title="Step 3", status=AutofixStatus.PROCESSING),
                ChangesStep(
                    id="4",
                    key="changes",
                    title="Step 4",
                    status=AutofixStatus.PROCESSING,
                    changes=[],
                ),
            ]
            cur.codebases = {
                "repo1": CodebaseState(
                    repo_id=1, repo_external_id="repo1", file_changes=[next(generate(FileChange))]
                )
            }

        event_manager.reset_steps_to_point(1, None)

        assert len(state.get().steps) == 2
        assert len(state.get().codebases["repo1"].file_changes) == 0

    def test_send_root_cause_analysis_start(self, event_manager, state):
        # No existing root cause step
        event_manager.send_root_cause_analysis_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 1
        root_cause_step = state_obj.steps[0]
        assert root_cause_step.key == event_manager.root_cause_analysis_processing_step.key
        assert root_cause_step.status == AutofixStatus.PROCESSING
        assert state_obj.status == AutofixStatus.PROCESSING

        #  Existing root cause step with different status
        with state.update() as cur:
            cur.steps[0].status = AutofixStatus.COMPLETED

        event_manager.send_root_cause_analysis_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 2  # Should add new step
        root_cause_step = state_obj.steps[-1]
        assert root_cause_step.key == event_manager.root_cause_analysis_processing_step.key
        assert root_cause_step.status == AutofixStatus.PROCESSING

        # Existing processing root cause step
        event_manager.send_root_cause_analysis_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 2  # Should not add new step
        root_cause_step = state_obj.steps[-1]
        assert root_cause_step.key == event_manager.root_cause_analysis_processing_step.key
        assert root_cause_step.status == AutofixStatus.PROCESSING

    def test_send_coding_start(self, event_manager, state):
        # No existing plan step
        event_manager.send_coding_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 1
        plan_step = state_obj.steps[0]
        assert plan_step.key == event_manager.plan_step.key
        assert plan_step.status == AutofixStatus.PROCESSING
        assert state_obj.status == AutofixStatus.PROCESSING

        # Existing plan step with different status
        with state.update() as cur:
            cur.steps[0].status = AutofixStatus.COMPLETED

        event_manager.send_coding_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 2  # Should add new step
        plan_step = state_obj.steps[-1]
        assert plan_step.key == event_manager.plan_step.key
        assert plan_step.status == AutofixStatus.PROCESSING

        # Existing processing plan step
        event_manager.send_coding_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 2  # Should not add new step
        plan_step = state_obj.steps[-1]
        assert plan_step.key == event_manager.plan_step.key
        assert plan_step.status == AutofixStatus.PROCESSING

    def test_ask_user_question(self, event_manager, state):
        # Setup initial state with a processing step
        state.get().steps = [
            RootCauseStep(id="1", title="Test Step", status=AutofixStatus.PROCESSING, progress=[])
        ]

        test_question = "Would you like to proceed?"
        event_manager.ask_user_question(test_question)

        # Check the current step's status
        assert state.get().steps[-1].status == AutofixStatus.WAITING_FOR_USER_RESPONSE

        # Check if the question was added to progress items
        assert len(state.get().steps[-1].progress) == 1
        assert state.get().steps[-1].progress[0].message == test_question
        assert state.get().steps[-1].progress[0].type == ProgressType.INFO

        # Check overall state status
        assert state.get().status == AutofixStatus.WAITING_FOR_USER_RESPONSE

    def test_send_solution_start(self, event_manager, state):
        # No existing solution step
        event_manager.send_solution_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 1
        solution_step = state_obj.steps[0]
        assert solution_step.key == event_manager.solution_processing_step.key
        assert solution_step.status == AutofixStatus.PROCESSING
        assert state_obj.status == AutofixStatus.PROCESSING

        # Existing solution step with different status
        with state.update() as cur:
            cur.steps[0].status = AutofixStatus.COMPLETED

        event_manager.send_solution_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 2  # Should add new step
        solution_step = state_obj.steps[-1]
        assert solution_step.key == event_manager.solution_processing_step.key
        assert solution_step.status == AutofixStatus.PROCESSING

        # Existing processing solution step
        event_manager.send_solution_start()

        state_obj = state.get()
        assert len(state_obj.steps) == 2  # Should not add new step
        solution_step = state_obj.steps[-1]
        assert solution_step.key == event_manager.solution_processing_step.key
        assert solution_step.status == AutofixStatus.PROCESSING

    def test_send_solution_result(self, event_manager, state):
        # Create a mock solution output
        mock_solution_output = next(generate(SolutionOutput))

        event_manager.send_solution_result(mock_solution_output)

        state_obj = state.get()
        # Check solution processing step
        solution_processing_step = next(
            (
                step
                for step in state_obj.steps
                if step.key == event_manager.solution_processing_step.key
            ),
            None,
        )
        assert solution_processing_step is not None
        assert solution_processing_step.status == AutofixStatus.COMPLETED

        # Check solution step
        solution_step = next(
            (step for step in state_obj.steps if step.key == event_manager.solution_step.key),
            None,
        )
        assert solution_step is not None
        assert solution_step.status == AutofixStatus.COMPLETED
        assert len(solution_step.solution) > 0
        assert state_obj.status == AutofixStatus.NEED_MORE_INFORMATION

    def test_set_selected_solution(self, event_manager, state):
        payload = AutofixSolutionUpdatePayload(
            solution=[
                SolutionTimelineEvent(
                    title="Custom solution",
                    timeline_item_type="human_instruction",
                )
            ]
        )

        event_manager.send_solution_result(SolutionOutput(solution_steps=[], summary=""))
        event_manager.set_selected_solution(payload)

        state_obj = state.get()
        solution_step = next(
            (step for step in state_obj.steps if step.key == event_manager.solution_step.key),
            None,
        )
        assert solution_step is not None
        assert solution_step.solution[0].title == "Custom solution"
        assert solution_step.solution_selected is True
        assert state_obj.status == AutofixStatus.PROCESSING

        # Test without custom solution
        payload = AutofixSolutionUpdatePayload(custom_solution=None)

        event_manager.set_selected_solution(payload)

        state_obj = state.get()
        solution_step = next(
            (step for step in state_obj.steps if step.key == event_manager.solution_step.key),
            None,
        )
        assert solution_step is not None
        assert solution_step.custom_solution is None
        assert solution_step.solution_selected is True
        assert state_obj.status == AutofixStatus.PROCESSING

        # Verify file changes are cleared
        with state.update() as cur:
            cur.codebases = {
                "repo1": CodebaseState(
                    repo_external_id="repo1", file_changes=[next(generate(FileChange))]
                )
            }

        event_manager.set_selected_solution(payload)
        assert len(state.get().codebases["repo1"].file_changes) == 0

        # Verify steps after solution step are deleted
        with state.update() as cur:
            cur.steps.append(DefaultStep(id="after_solution", title="After Solution"))

        event_manager.set_selected_solution(payload)
        assert len(state.get().steps) == 2
        assert state.get().steps[0].key == event_manager.solution_processing_step.key
        assert state.get().steps[1].key == event_manager.solution_step.key
