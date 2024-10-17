import datetime
from unittest.mock import MagicMock, call, patch

import pytest
from johen import generate
from unidiff import Hunk

from seer.automation.agent.models import Message, ToolCall
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixCreatePrUpdatePayload,
    AutofixRequest,
    AutofixRestartFromPointPayload,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    AutofixUpdateCodeChangePayload,
    AutofixUpdateRequest,
    AutofixUpdateType,
    AutofixUserMessagePayload,
    ChangesStep,
    DefaultStep,
    Step,
)
from seer.automation.autofix.steps.coding_step import AutofixCodingStep, AutofixCodingStepRequest
from seer.automation.autofix.tasks import (
    check_and_mark_recent_autofix_runs,
    get_autofix_state,
    get_autofix_state_from_pr_id,
    receive_user_message,
    restart_from_point_with_feedback,
    restart_step_with_user_response,
    run_autofix_create_pr,
    run_autofix_execution,
    run_autofix_root_cause,
    truncate_memory_to_match_insights,
    update_code_change,
)
from seer.automation.models import FilePatch, Line
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session


class TestGetAutofixState:
    def test_get_state_by_group_id(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
            session.commit()

        retrieved_state = get_autofix_state(group_id=100)
        assert retrieved_state is not None
        if retrieved_state is not None:
            assert retrieved_state.get() == state

    def test_get_state_by_run_id(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=2, group_id=200, value=state.model_dump(mode="json")))
            session.commit()

        retrieved_state = get_autofix_state(run_id=2)
        assert retrieved_state is not None
        if retrieved_state is not None:
            assert retrieved_state.get() == state

    def test_get_state_no_matching_group_id(self):
        retrieved_state = get_autofix_state(group_id=999)
        assert retrieved_state is None

    def test_get_state_no_matching_run_id(self):
        retrieved_state = get_autofix_state(run_id=999)
        assert retrieved_state is None

    def test_get_state_multiple_runs_for_group(self):
        states = [next(generate(AutofixContinuation)) for _ in range(3)]
        with Session() as session:
            for i, state in enumerate(states, start=1):
                session.add(DbRunState(id=i, group_id=300, value=state.model_dump(mode="json")))
            session.commit()

        retrieved_state = get_autofix_state(group_id=300)
        assert retrieved_state is not None
        if retrieved_state is not None:
            # Should return the most recent state (highest id)
            assert retrieved_state.get() == states[-1]

    def test_get_state_no_parameters(self):
        with pytest.raises(ValueError, match="Either group_id or run_id must be provided"):
            get_autofix_state()

    def test_get_state_both_parameters(self):
        with pytest.raises(
            ValueError, match="Either group_id or run_id must be provided, not both"
        ):
            get_autofix_state(group_id=1, run_id=1)


class TestGetStateFromPr:
    def test_successful_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 1)
        assert retrieved_state is not None
        if retrieved_state is not None:
            assert retrieved_state.get() == state

    def test_no_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 2)
        assert retrieved_state is None


class TestRunAutofixRootCause:
    @patch("seer.automation.autofix.tasks.create_initial_autofix_run")
    @patch("seer.automation.autofix.tasks.RootCauseStep")
    def test_happy_path(self, mock_root_cause_step, mock_create_initial_autofix_run):
        # Setup
        mock_request = MagicMock(spec=AutofixRequest)
        mock_state = MagicMock()
        mock_state.get.return_value = MagicMock(run_id=1, status=AutofixStatus.PROCESSING)
        mock_create_initial_autofix_run.return_value = mock_state

        mock_signature = MagicMock()
        mock_root_cause_step.get_signature.return_value = mock_signature

        # Execute
        result = run_autofix_root_cause(mock_request)

        # Assert
        mock_create_initial_autofix_run.assert_called_once_with(mock_request)
        mock_root_cause_step.get_signature.assert_called_once()
        assert mock_root_cause_step.get_signature.call_args[0][0].run_id == 1
        assert isinstance(mock_root_cause_step.get_signature.call_args[0][0].step_id, int)
        mock_signature.apply_async.assert_called_once()
        assert result == 1


class TestRunAutofixExecution:
    @patch("seer.automation.autofix.tasks.ContinuationState")
    @patch("seer.automation.autofix.tasks.AutofixEventManager")
    @patch("seer.automation.autofix.tasks.AutofixCodingStep")
    def test_happy_path(self, mock_coding_step, mock_event_manager, mock_continuation_state):
        # Setup
        mock_request = MagicMock(spec=AutofixUpdateRequest, run_id=1)
        mock_request.payload = MagicMock(spec=AutofixRootCauseUpdatePayload)

        mock_state = MagicMock()
        mock_state.get.return_value = MagicMock(run_id=1, status=AutofixStatus.PROCESSING)
        mock_continuation_state.from_id.return_value = mock_state

        mock_signature = MagicMock()
        mock_coding_step.get_signature.return_value = mock_signature

        # Execute
        run_autofix_execution(mock_request)

        # Assert
        mock_continuation_state.from_id.assert_called_once_with(1, model=AutofixContinuation)
        mock_event_manager.return_value.send_coding_start.assert_called_once()
        mock_event_manager.return_value.set_selected_root_cause.assert_called_once_with(
            mock_request.payload
        )
        mock_coding_step.get_signature.assert_called_once()
        assert mock_coding_step.get_signature.call_args[0][0].run_id == 1
        assert isinstance(mock_coding_step.get_signature.call_args[0][0].step_id, int)
        mock_signature.apply_async.assert_called_once()


class TestRunAutofixCreatePr:
    @patch("seer.automation.autofix.tasks.ContinuationState")
    @patch("seer.automation.autofix.tasks.AutofixEventManager")
    @patch("seer.automation.autofix.tasks.AutofixContext")
    def test_happy_path(self, mock_autofix_context, mock_event_manager, mock_continuation_state):
        # Setup
        mock_request = MagicMock(spec=AutofixUpdateRequest, run_id=1)
        mock_request.payload = MagicMock(
            spec=AutofixCreatePrUpdatePayload, repo_external_id="repo1", repo_id=1
        )

        mock_state = MagicMock()
        mock_continuation_state.from_id.return_value = mock_state

        mock_context = MagicMock()
        mock_autofix_context.return_value = mock_context

        # Execute
        run_autofix_create_pr(mock_request)

        # Assert
        mock_continuation_state.from_id.assert_called_once_with(1, model=AutofixContinuation)
        mock_context.commit_changes.assert_called_once_with(repo_external_id="repo1", repo_id=1)


class TestCheckAndMarkRecentAutofixRuns:
    @patch("seer.automation.autofix.tasks.datetime")
    @patch("seer.automation.autofix.tasks.get_all_autofix_runs_after")
    @patch("seer.automation.autofix.tasks.check_and_mark_if_timed_out")
    @patch("seer.automation.autofix.tasks.logger")
    def test_check_and_mark_recent_autofix_runs(
        self, mock_logger, mock_check_and_mark, mock_get_runs, mock_datetime
    ):
        # Setup
        mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.datetime.now.return_value = mock_now
        mock_one_hour_ago = mock_now - datetime.timedelta(hours=1)
        mock_datetime.timedelta.return_value = datetime.timedelta(hours=1)

        mock_run1 = MagicMock()
        mock_run2 = MagicMock()
        mock_get_runs.return_value = [mock_run1, mock_run2]

        # Execute
        check_and_mark_recent_autofix_runs()

        # Assert
        mock_datetime.datetime.now.assert_called_once()
        mock_datetime.timedelta.assert_called_once_with(hours=1, minutes=15)
        mock_get_runs.assert_called_once_with(mock_one_hour_ago)
        mock_logger.info.assert_any_call("Checking and marking recent autofix runs")
        mock_logger.info.assert_any_call(f"Getting all autofix runs after {mock_one_hour_ago}")
        mock_logger.info.assert_any_call("Got 2 runs")
        mock_check_and_mark.assert_has_calls([call(mock_run1), call(mock_run2)])
        assert mock_check_and_mark.call_count == 2


class TestHandleUserMessages:
    @pytest.fixture
    def mock_continuation_state(self):
        with patch("seer.automation.autofix.tasks.ContinuationState") as mock_cs:
            yield mock_cs

    @pytest.fixture
    def mock_event_manager(self):
        with patch("seer.automation.autofix.tasks.AutofixEventManager") as mock_em:
            yield mock_em

    @pytest.fixture
    def mock_context(self):
        with patch("seer.automation.autofix.tasks.AutofixContext") as mock_ctx:
            yield mock_ctx

    def test_receive_user_message_response_to_question(self, mock_continuation_state, mock_context):
        mock_payload = AutofixUserMessagePayload(
            type=AutofixUpdateType.USER_MESSAGE, text="User response"
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_state.get.return_value.find_last_step_waiting_for_response.return_value = MagicMock()

        mock_coding_memory = [
            Message(
                tool_calls=[ToolCall(function="func", args='{"question": "Test Question"}')],
                tool_call_id="tool_1",
            ),
            Message(tool_call_id="tool_1"),
        ]
        mock_context.return_value.get_memory.return_value = mock_coding_memory

        mock_step = MagicMock(spec=DefaultStep)
        mock_step.insights = []
        mock_state.update.return_value.__enter__.return_value.steps = [mock_step]

        receive_user_message(mock_request)

        mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
        mock_state.update.assert_called()
        mock_context.assert_called()
        assert "Test Question" in mock_step.insights[-1].insight
        assert mock_step.insights[-1].insight.endswith("User response")

    def test_receive_user_message_interjection(self, mock_continuation_state):
        mock_payload = AutofixUserMessagePayload(
            type=AutofixUpdateType.USER_MESSAGE, text="User interjection"
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_state.get.return_value.find_last_step_waiting_for_response.return_value = None
        mock_step = MagicMock(spec=DefaultStep)
        mock_step.insights = []
        mock_state.update.return_value.__enter__.return_value.steps = [mock_step]

        receive_user_message(mock_request)

        mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
        mock_step.receive_user_message.assert_called_once_with("User interjection")
        assert isinstance(mock_step.insights[-1], InsightSharingOutput)
        assert mock_step.insights[-1].insight == "User interjection"
        assert mock_step.insights[-1].justification == "USER"

    def test_receive_user_message_invalid_payload_type(self):
        mock_payload = AutofixRootCauseUpdatePayload(
            type=AutofixUpdateType.SELECT_ROOT_CAUSE, text="Invalid payload type"
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        with pytest.raises(ValueError, match="Invalid payload type for user_message"):
            receive_user_message(mock_request)

    def test_restart_step_with_user_response(self, mock_event_manager):
        mock_state = MagicMock()
        mock_state.get.return_value.run_id = 123
        mock_memory = [Message(role="assistant", content="Previous message", tool_call_id="tool_1")]
        mock_step_to_restart = MagicMock(spec=Step)
        mock_step_class = MagicMock(spec=AutofixCodingStep)
        mock_step_request_class = MagicMock(spec=AutofixCodingStepRequest)

        restart_step_with_user_response(
            mock_state,
            mock_memory,
            "User response",
            mock_event_manager,
            mock_step_to_restart,
            mock_step_class,
            mock_step_request_class,
        )

        assert len(mock_memory) == 2
        assert mock_memory[-1].role == "tool"
        assert mock_memory[-1].content == "User response"
        assert mock_memory[-1].tool_call_id == "tool_1"

        mock_event_manager.restart_step.assert_called_once_with(mock_step_to_restart)

        mock_step_class.get_signature.assert_called_once()
        mock_step_request_class.assert_called_once_with(
            run_id=123,
            initial_memory=mock_memory,
        )
        mock_step_class.get_signature.return_value.apply_async.assert_called_once_with()

    def test_restart_from_point_with_feedback_invalid_payload(self):
        mock_payload = AutofixUserMessagePayload(
            type=AutofixUpdateType.USER_MESSAGE, text="Invalid payload"
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        with pytest.raises(
            ValueError, match="Invalid payload type for restart_from_point_with_feedback"
        ):
            restart_from_point_with_feedback(mock_request)

    def test_restart_from_point_with_feedback_coding_step(
        self, mock_continuation_state, mock_event_manager, mock_context
    ):
        with patch("seer.automation.autofix.tasks.AutofixCodingStep") as mock_coding_step:
            mock_payload = AutofixRestartFromPointPayload(
                type=AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK,
                step_index=1,
                retain_insight_card_index=0,
                message="User feedback",
            )
            mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

            mock_state = mock_continuation_state.from_id.return_value
            mock_step = MagicMock(spec=DefaultStep)
            mock_step.key = "plan"
            mock_step.insights = [
                MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
                MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
                MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
            ]
            mock_steps = [mock_step]
            mock_state.get.return_value.steps = mock_steps
            mock_state.get.return_value.run_id = 123
            mock_state.update.return_value.__enter__.return_value.find_step.side_effect = (
                lambda index: mock_steps[index]
            )
            mock_state.update.return_value.__enter__.return_value.steps = mock_steps

            mock_context.return_value.get_memory.return_value = [
                Message(content="Previous message", role="assistant"),
                Message(content="User message", role="user"),
            ]

            restart_from_point_with_feedback(mock_request)

            mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
            mock_state.update.assert_called()
            mock_event_manager.return_value.reset_steps_to_point.assert_called_once_with(1, 0)
            mock_event_manager.return_value.restart_step.assert_called_once_with(mock_step)

            mock_coding_step.get_signature.assert_called_once()
            mock_coding_step.get_signature.return_value.apply_async.assert_called_once()

    def test_restart_from_point_with_feedback_root_cause_step(
        self, mock_continuation_state, mock_event_manager, mock_context
    ):
        mock_payload = AutofixRestartFromPointPayload(
            type=AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK,
            step_index=0,
            retain_insight_card_index=None,
            message="User feedback",
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_step = MagicMock(spec=DefaultStep)
        mock_step.key = "root_cause_analysis_processing"
        mock_step.insights = []
        mock_state.get.return_value.steps = [mock_step]
        mock_state.get.return_value.run_id = 123
        mock_state.update.return_value.__enter__.return_value.steps = [mock_step]

        mock_context.return_value.get_memory.return_value = [
            Message(content="Initial message", role="system")
        ]

        restart_from_point_with_feedback(mock_request)

        mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
        mock_state.update.assert_called()
        mock_event_manager.return_value.restart_step.assert_called_once_with(mock_step)

        assert len(mock_step.insights) == 1
        assert isinstance(mock_step.insights[-1], InsightSharingOutput)
        assert mock_step.insights[-1].insight == "User feedback"
        assert mock_step.insights[-1].justification == "USER"

    def test_truncate_memory_to_match_insights(self):
        # Create a mock step with insights
        mock_step = MagicMock(spec=DefaultStep)
        mock_step.insights = [
            MagicMock(spec=InsightSharingOutput, generated_at_memory_index=2),
            MagicMock(spec=InsightSharingOutput, generated_at_memory_index=4),
            MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
        ]

        # Create a mock memory list
        memory = [
            Message(content="Initial message", role="system"),
            Message(content="User message 1", role="user"),
            Message(content="Assistant response 1", role="assistant"),
            Message(content="User message 2", role="user"),
            Message(
                content="Assistant response 2",
                role="assistant",
                tool_calls=[
                    ToolCall(function="func1", args="{}"),
                    ToolCall(function="func2", args="{}"),
                ],
            ),
            Message(content="Tool response 1", role="tool", tool_call_id="func1"),
            Message(content="Tool response 2", role="tool", tool_call_id="func2"),
            Message(content="Final message", role="assistant"),
        ]

        # Test case 1: Normal truncation
        truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
        assert len(truncated_memory) == 7  # Up to and including the tool responses
        assert truncated_memory[-1].role == "tool"
        assert truncated_memory[-1].content == "Tool response 2"

        # Test case 2: No insights
        mock_step.insights = []
        truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
        assert len(truncated_memory) == 1
        assert truncated_memory[0].content == "Initial message"

        # Test case 3: All insights have negative index
        mock_step.insights = [
            MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
            MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
        ]
        truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
        assert len(truncated_memory) == len(memory)
        assert truncated_memory[0].content == "Initial message"

        # Test case 4: Truncation without tool calls
        mock_step.insights = [MagicMock(spec=InsightSharingOutput, generated_at_memory_index=2)]
        truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
        assert len(truncated_memory) == 3
        assert truncated_memory[-1].content == "Assistant response 1"

        # Test case 5: Truncation with incomplete tool responses
        mock_step.insights = [MagicMock(spec=InsightSharingOutput, generated_at_memory_index=4)]
        memory_incomplete = memory[:6]  # Remove the last tool response
        truncated_memory = truncate_memory_to_match_insights(memory_incomplete, mock_step)
        assert (
            len(truncated_memory) == 4
        )  # Up to Assistant response 2, excluding incomplete tool calls
        assert truncated_memory[-1].content == "User message 2"


class TestUpdateCodeChange:
    @pytest.fixture
    def mock_continuation_state(self):
        with patch("seer.automation.autofix.tasks.ContinuationState") as mock_cs:
            yield mock_cs

    def test_update_code_change_happy_path(self, mock_continuation_state):
        # Setup
        mock_payload = AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_id="repo1",
            hunk_index=0,
            lines=[
                Line(line_type=" ", value="unchanged line"),
                Line(line_type="+", value="new line"),
                Line(line_type="-", value="removed line"),
            ],
            file_path="path/to/file.py",
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_cur_state = MagicMock()
        mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
        mock_state.get.return_value = mock_cur_state

        mock_change = MagicMock()
        mock_change.repo_external_id = "repo1"
        mock_file_patch = MagicMock(spec=FilePatch)
        mock_file_patch.path = "path/to/file.py"
        mock_hunk = MagicMock(spec=Hunk)
        mock_hunk.target_start = 10
        mock_hunk.target_length = 1
        mock_file_patch.hunks = [mock_hunk]
        mock_change.diff = [mock_file_patch]
        mock_cur_state.steps[-1].changes = [mock_change]

        # Execute
        update_code_change(mock_request)

        # Assert
        mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
        mock_state.update.assert_called_once()

        # Check if the hunk was updated correctly
        updated_hunk = mock_file_patch.hunks[0]
        assert updated_hunk.lines == mock_payload.lines
        assert updated_hunk.target_length == 2  # 1 unchanged + 1 new line

        # Check if subsequent hunks were updated (if any)
        if len(mock_file_patch.hunks) > 1:
            assert mock_file_patch.hunks[1].target_start == 12  # 10 + 2

    def test_update_code_change_no_matching_change(self, mock_continuation_state):
        mock_payload = AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_id="non_existent_repo",
            hunk_index=0,
            lines=[],
            file_path="path/to/file.py",
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_cur_state = MagicMock()
        mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
        mock_state.get.return_value = mock_cur_state
        mock_cur_state.steps[-1].changes = []

        with pytest.raises(ValueError, match="No matching change found"):
            update_code_change(mock_request)

    def test_update_code_change_no_matching_file_patch(self, mock_continuation_state):
        mock_payload = AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_id="repo1",
            hunk_index=0,
            lines=[],
            file_path="non_existent_file.py",
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_cur_state = MagicMock()
        mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
        mock_state.get.return_value = mock_cur_state

        mock_change = MagicMock()
        mock_change.repo_external_id = "repo1"
        mock_change.diff = []
        mock_cur_state.steps[-1].changes = [mock_change]

        with pytest.raises(ValueError, match="No matching file patch found"):
            update_code_change(mock_request)

    def test_update_code_change_invalid_hunk_index(self, mock_continuation_state):
        mock_payload = AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_id="repo1",
            hunk_index=99,  # Invalid index
            lines=[],
            file_path="path/to/file.py",
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_cur_state = MagicMock()
        mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
        mock_state.get.return_value = mock_cur_state

        mock_change = MagicMock()
        mock_change.repo_external_id = "repo1"
        mock_file_patch = MagicMock(spec=FilePatch)
        mock_file_patch.path = "path/to/file.py"
        mock_file_patch.hunks = []
        mock_change.diff = [mock_file_patch]
        mock_cur_state.steps[-1].changes = [mock_change]

        with pytest.raises(ValueError, match="Hunk index is out of range"):
            update_code_change(mock_request)

    def test_update_code_change_invalid_step_type(self, mock_continuation_state):
        mock_payload = AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_id="repo1",
            hunk_index=0,
            lines=[],
            file_path="path/to/file.py",
        )
        mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)

        mock_state = mock_continuation_state.from_id.return_value
        mock_cur_state = MagicMock()
        mock_cur_state.steps = [MagicMock(spec=DefaultStep)]  # Not a ChangesStep
        mock_state.get.return_value = mock_cur_state

        # Execute
        update_code_change(mock_request)

        # Assert
        mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
        mock_state.update.assert_not_called()  # The function should return early without updating
