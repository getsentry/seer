import datetime
from typing import cast

import pytest
from johen import generate

from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixCreatePrUpdatePayload,
    AutofixRequest,
    AutofixRestartFromPointPayload,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    AutofixUpdateRequest,
    AutofixUpdateType,
    ChangesStep,
    DefaultStep,
    RootCauseStep,
)
from seer.automation.autofix.tasks import (
    check_and_mark_recent_autofix_runs,
    get_autofix_state,
    get_autofix_state_from_pr_id,
    restart_from_point_with_feedback,
    run_autofix_execution,
    run_autofix_push_changes,
    run_autofix_root_cause,
)
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session
from seer.dependency_injection import resolve
from seer.rpc import DummyRpcClient, RpcClient
from tests.utils import eager_celery


@pytest.fixture(autouse=True)
def setup_rpc_client():
    rpc_client = resolve(RpcClient)
    if isinstance(rpc_client, DummyRpcClient):
        rpc_client.dry_run = True


@pytest.fixture
def autofix_request():
    with open("tests/data/autofix_request.json") as f:
        return AutofixRequest.model_validate_json(f.read())


@pytest.fixture
def autofix_full_finished_run():
    with open("tests/data/autofix_full_finished_run.json") as f:
        return AutofixContinuation.model_validate_json(f.read())


@pytest.fixture
def autofix_root_cause_run():
    with open("tests/data/autofix_root_cause_run.json") as f:
        return AutofixContinuation.model_validate_json(f.read())


def test_get_state_by_group_id():
    state = next(generate(AutofixContinuation))
    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    retrieved_state = get_autofix_state(group_id=100)
    assert retrieved_state is not None
    if retrieved_state is not None:
        assert retrieved_state.get() == state


def test_get_state_by_run_id():
    state = next(generate(AutofixContinuation))
    with Session() as session:
        session.add(DbRunState(id=2, group_id=200, value=state.model_dump(mode="json")))
        session.commit()

    retrieved_state = get_autofix_state(run_id=2)
    assert retrieved_state is not None
    if retrieved_state is not None:
        assert retrieved_state.get() == state


def test_get_state_no_matching_group_id():
    retrieved_state = get_autofix_state(group_id=999)
    assert retrieved_state is None


def test_get_state_no_matching_run_id():
    retrieved_state = get_autofix_state(run_id=999)
    assert retrieved_state is None


def test_get_state_multiple_runs_for_group():
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


def test_get_state_no_parameters():
    with pytest.raises(ValueError, match="Either group_id or run_id must be provided"):
        get_autofix_state()


def test_get_state_both_parameters():
    with pytest.raises(ValueError, match="Either group_id or run_id must be provided, not both"):
        get_autofix_state(group_id=1, run_id=1)


def test_successful_state_mapping():
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


def test_no_state_mapping():
    state = next(generate(AutofixContinuation))
    with Session() as session:
        session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
        session.flush()
        session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
        session.commit()

    retrieved_state = get_autofix_state_from_pr_id("test", 2)
    assert retrieved_state is None


@pytest.mark.vcr()
def test_autofix_run_root_cause_analysis(autofix_request: AutofixRequest):
    with eager_celery():
        run_id = run_autofix_root_cause(autofix_request)

    assert run_id is not None

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    root_cause_step = continuation.get().find_step(key="root_cause_analysis")

    assert root_cause_step is not None
    root_cause_step = cast(RootCauseStep, root_cause_step)
    assert root_cause_step.status == AutofixStatus.COMPLETED
    assert root_cause_step.causes is not None
    assert len(root_cause_step.causes) > 0

    assert continuation.get().status not in {AutofixStatus.ERROR}


@pytest.mark.vcr()
def test_autofix_run_full(autofix_request: AutofixRequest):
    with eager_celery():
        run_id = run_autofix_root_cause(autofix_request)

    assert run_id is not None

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    root_cause_step = continuation.get().find_step(key="root_cause_analysis")

    assert root_cause_step is not None
    root_cause_step = cast(RootCauseStep, root_cause_step)
    assert root_cause_step.status == AutofixStatus.COMPLETED

    with eager_celery():
        run_autofix_execution(
            AutofixUpdateRequest(
                run_id=run_id,
                payload=AutofixRootCauseUpdatePayload(
                    custom_root_cause="we should uncomment out the unit test parts"
                ),
            )
        )

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    changes_step = continuation.get().find_step(key="changes")

    assert changes_step is not None
    changes_step = cast(ChangesStep, changes_step)
    assert changes_step.status == AutofixStatus.COMPLETED
    assert len(changes_step.changes) > 0

    assert continuation.get().status not in {AutofixStatus.ERROR}


@pytest.mark.vcr()
def test_autofix_run_question_asking(autofix_request: AutofixRequest):
    autofix_request.instruction = "The root cause of this is not clear and you must think it through. You MUST use the question asking tool and ask a question."
    autofix_request.issue.events[0][
        "title"
    ] = "The root cause of this is not clear and you must search the codebase for the hidden answer"

    with eager_celery():
        run_id = run_autofix_root_cause(autofix_request)

    assert run_id is not None

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    root_cause_processing_step = continuation.get().find_step(key="root_cause_analysis_processing")

    assert root_cause_processing_step is not None
    root_cause_processing_step = cast(DefaultStep, root_cause_processing_step)
    assert root_cause_processing_step.status == AutofixStatus.WAITING_FOR_USER_RESPONSE
    assert root_cause_processing_step.insights is not None


@pytest.mark.vcr()
def test_autofix_run_coding(autofix_root_cause_run: AutofixContinuation):
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_root_cause_run.run_id,
                group_id=1,
                value=autofix_root_cause_run.model_dump(mode="json"),
            )
        )
        session.commit()

    with eager_celery():
        run_autofix_execution(
            AutofixUpdateRequest(
                run_id=autofix_root_cause_run.run_id,
                payload=AutofixRootCauseUpdatePayload(
                    custom_root_cause="we should uncomment out the unit test parts"
                ),
            )
        )

    continuation = get_autofix_state(run_id=autofix_root_cause_run.run_id)

    assert continuation is not None

    changes_step = continuation.get().find_step(key="changes")

    assert changes_step is not None
    changes_step = cast(ChangesStep, changes_step)
    assert changes_step.status == AutofixStatus.COMPLETED
    assert len(changes_step.changes) > 0

    assert continuation.get().status not in {AutofixStatus.ERROR}


@pytest.mark.vcr()
def test_autofix_restart_from_point_with_feedback(autofix_root_cause_run: AutofixContinuation):
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_root_cause_run.run_id,
                group_id=1,
                value=autofix_root_cause_run.model_dump(mode="json"),
            )
        )

        session.commit()

    context = AutofixContext.from_run_id(autofix_root_cause_run.run_id)
    context.store_memory(
        "root_cause_analysis",
        [
            Message(content="This is message 1", role="user"),
            Message(content="This is message 2", role="assistant"),
        ],
    )

    with eager_celery():
        restart_from_point_with_feedback(
            AutofixUpdateRequest(
                run_id=autofix_root_cause_run.run_id,
                payload=AutofixRestartFromPointPayload(
                    type=AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK,
                    step_index=0,
                    retain_insight_card_index=0,
                    message="we should uncomment out the unit test parts",
                ),
            )
        )

    with Session() as session:
        new_run_memory = context.get_memory("root_cause_analysis")

        assert new_run_memory[0].content == "This is message 1"
        assert new_run_memory[1].content == "."
        assert new_run_memory[2].content == "we should uncomment out the unit test parts"

    continuation = get_autofix_state(run_id=autofix_root_cause_run.run_id)

    assert continuation is not None

    root_cause_step = continuation.get().find_step(key="root_cause_analysis")

    assert root_cause_step is not None
    root_cause_step = cast(RootCauseStep, root_cause_step)
    assert root_cause_step.status == AutofixStatus.COMPLETED
    assert root_cause_step.causes is not None
    assert len(root_cause_step.causes) > 0

    assert continuation.get().status not in {AutofixStatus.ERROR}


@pytest.mark.vcr()
def test_autofix_create_pr(autofix_full_finished_run: AutofixContinuation):
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_full_finished_run.run_id,
                group_id=1,
                value=autofix_full_finished_run.model_dump(mode="json"),
            )
        )
        session.commit()

    repo_external_id = next(iter(autofix_full_finished_run.codebases.keys()))

    with eager_celery():
        run_autofix_push_changes(
            AutofixUpdateRequest(
                run_id=autofix_full_finished_run.run_id,
                payload=AutofixCreatePrUpdatePayload(repo_external_id=repo_external_id),
            ),
        )

    with Session() as session:
        pr_id_to_run_id_mapping = (
            session.query(DbPrIdToAutofixRunIdMapping)
            .filter_by(run_id=autofix_full_finished_run.run_id)
            .first()
        )
        assert pr_id_to_run_id_mapping is not None
        assert pr_id_to_run_id_mapping.pr_id is not None


def make_continuation(
    run_id: int, last_triggered_at: datetime.datetime, updated_at: datetime.datetime
):
    continuation = next(generate(AutofixContinuation))
    continuation.run_id = run_id
    continuation.last_triggered_at = last_triggered_at
    continuation.updated_at = updated_at
    return continuation.model_dump(mode="json")


def test_check_and_mark_recent_autofix_runs():
    # Get current time for creating relative timestamps
    now = datetime.datetime.now()

    # Create test runs with different states and timestamps
    runs_data = [
        # Run that should timeout due to no updates in 90 seconds
        {
            "id": 1,
            "group_id": 100,
            "value": make_continuation(
                1, now - datetime.timedelta(minutes=5), now - datetime.timedelta(seconds=91)
            ),
            "last_triggered_at": now - datetime.timedelta(minutes=5),
            "updated_at": now - datetime.timedelta(seconds=91),
        },
        # Run that should timeout due to running for over 10 minutes
        {
            "id": 2,
            "group_id": 200,
            "value": make_continuation(
                2, now - datetime.timedelta(minutes=11), now - datetime.timedelta(seconds=30)
            ),
            "last_triggered_at": now - datetime.timedelta(minutes=11),
            "updated_at": now - datetime.timedelta(seconds=30),
        },
        # Run that should not timeout (recent activity)
        {
            "id": 3,
            "group_id": 300,
            "value": make_continuation(
                3, now - datetime.timedelta(minutes=5), now - datetime.timedelta(seconds=30)
            ),
            "last_triggered_at": now - datetime.timedelta(minutes=5),
            "updated_at": now - datetime.timedelta(seconds=30),
        },
    ]

    # Insert runs into database
    with Session() as session:
        for run_data in runs_data:
            run_state = DbRunState(**run_data)
            # Set the status to PROCESSING for each run
            state_dict = run_state.value
            state_dict["status"] = "PROCESSING"
            run_state.value = state_dict
            session.add(run_state)
        session.commit()

    # Execute
    check_and_mark_recent_autofix_runs()

    # Assert - Check the status of each run
    with Session() as session:
        # Run 1 should be marked as ERROR due to no recent updates
        run1 = session.query(DbRunState).filter_by(id=1).first()
        assert run1 is not None
        assert run1.value["status"] == "ERROR"

        # Run 2 should be marked as ERROR due to running too long
        run2 = session.query(DbRunState).filter_by(id=2).first()
        assert run2 is not None
        assert run2.value["status"] == "ERROR"

        # Run 3 should still be PROCESSING
        run3 = session.query(DbRunState).filter_by(id=3).first()
        assert run3 is not None
        assert run3.value["status"] == "PROCESSING"


# class TestHandleUserMessages:
#     @pytest.fixture
#     def mock_continuation_state(self):
#         with patch("seer.automation.autofix.tasks.ContinuationState") as mock_cs:
#             yield mock_cs
#
#     @pytest.fixture
#     def mock_event_manager(self):
#         with patch("seer.automation.autofix.tasks.AutofixEventManager") as mock_em:
#             yield mock_em
#
#     @pytest.fixture
#     def mock_context(self):
#         with patch("seer.automation.autofix.tasks.AutofixContext") as mock_ctx:
#             yield mock_ctx
#
#     def test_receive_user_message_response_to_question(self, mock_continuation_state, mock_context):
#         mock_payload = AutofixUserMessagePayload(
#             type=AutofixUpdateType.USER_MESSAGE, text="User response"
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_state.get.return_value.find_last_step_waiting_for_response.return_value = MagicMock()
#
#         mock_coding_memory = [
#             Message(
#                 tool_calls=[ToolCall(function="func", args='{"question": "Test Question"}')],
#                 tool_call_id="tool_1",
#             ),
#             Message(tool_call_id="tool_1"),
#         ]
#         mock_context.return_value.get_memory.return_value = mock_coding_memory
#
#         mock_step = MagicMock(spec=DefaultStep)
#         mock_step.insights = []
#         mock_state.update.return_value.__enter__.return_value.steps = [mock_step]
#
#         receive_user_message(mock_request)
#
#         mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
#         mock_state.update.assert_called()
#         mock_context.assert_called()
#         assert "Test Question" in mock_step.insights[-1].insight
#         assert mock_step.insights[-1].insight.endswith("User response")
#
#     def test_receive_user_message_interjection(self, mock_continuation_state):
#         mock_payload = AutofixUserMessagePayload(
#             type=AutofixUpdateType.USER_MESSAGE, text="User interjection"
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_state.get.return_value.find_last_step_waiting_for_response.return_value = None
#         mock_step = MagicMock(spec=DefaultStep)
#         mock_step.insights = []
#         mock_state.update.return_value.__enter__.return_value.steps = [mock_step]
#
#         receive_user_message(mock_request)
#
#         mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
#         mock_step.receive_user_message.assert_called_once_with("User interjection")
#         assert isinstance(mock_step.insights[-1], InsightSharingOutput)
#         assert mock_step.insights[-1].insight == "User interjection"
#         assert mock_step.insights[-1].justification == "USER"
#
#     def test_receive_user_message_invalid_payload_type(self):
#         mock_payload = AutofixRootCauseUpdatePayload(
#             type=AutofixUpdateType.SELECT_ROOT_CAUSE, text="Invalid payload type"
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         with pytest.raises(ValueError, match="Invalid payload type for user_message"):
#             receive_user_message(mock_request)
#
#     def test_restart_step_with_user_response(self, mock_event_manager):
#         mock_state = MagicMock()
#         mock_state.get.return_value.run_id = 123
#         mock_memory = [Message(role="assistant", content="Previous message", tool_call_id="tool_1")]
#         mock_step_to_restart = MagicMock(spec=Step)
#         mock_step_class = MagicMock(spec=AutofixCodingStep)
#         mock_step_request_class = MagicMock(spec=AutofixCodingStepRequest)
#
#         restart_step_with_user_response(
#             mock_state,
#             mock_memory,
#             "User response",
#             mock_event_manager,
#             mock_step_to_restart,
#             mock_step_class,
#             mock_step_request_class,
#         )
#
#         assert len(mock_memory) == 2
#         assert mock_memory[-1].role == "tool"
#         assert mock_memory[-1].content == "User response"
#         assert mock_memory[-1].tool_call_id == "tool_1"
#
#         mock_event_manager.restart_step.assert_called_once_with(mock_step_to_restart)
#
#         mock_step_class.get_signature.assert_called_once()
#         mock_step_request_class.assert_called_once_with(
#             run_id=123,
#             initial_memory=mock_memory,
#         )
#         mock_step_class.get_signature.return_value.apply_async.assert_called_once_with()
#
#     def test_restart_from_point_with_feedback_invalid_payload(self):
#         mock_payload = AutofixUserMessagePayload(
#             type=AutofixUpdateType.USER_MESSAGE, text="Invalid payload"
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         with pytest.raises(
#             ValueError, match="Invalid payload type for restart_from_point_with_feedback"
#         ):
#             restart_from_point_with_feedback(mock_request)
#
#     def test_restart_from_point_with_feedback_coding_step(
#         self, mock_continuation_state, mock_event_manager, mock_context
#     ):
#         with patch("seer.automation.autofix.tasks.AutofixCodingStep") as mock_coding_step:
#             mock_payload = AutofixRestartFromPointPayload(
#                 type=AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK,
#                 step_index=1,
#                 retain_insight_card_index=0,
#                 message="User feedback",
#             )
#             mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#             mock_state = mock_continuation_state.from_id.return_value
#             mock_step = MagicMock(spec=DefaultStep)
#             mock_step.key = "plan"
#             mock_step.insights = [
#                 MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
#                 MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
#                 MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
#             ]
#             mock_steps = [mock_step]
#             mock_state.get.return_value.steps = mock_steps
#             mock_state.get.return_value.run_id = 123
#             mock_state.update.return_value.__enter__.return_value.find_step.side_effect = (
#                 lambda index: mock_steps[index]
#             )
#             mock_state.update.return_value.__enter__.return_value.steps = mock_steps
#
#             mock_context.return_value.get_memory.return_value = [
#                 Message(content="Previous message", role="assistant"),
#                 Message(content="User message", role="user"),
#             ]
#
#             restart_from_point_with_feedback(mock_request)
#
#             mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
#             mock_state.update.assert_called()
#             mock_event_manager.return_value.reset_steps_to_point.assert_called_once_with(1, 0)
#             mock_event_manager.return_value.restart_step.assert_called_once_with(mock_step)
#
#             mock_coding_step.get_signature.assert_called_once()
#             mock_coding_step.get_signature.return_value.apply_async.assert_called_once()
#
#     def test_restart_from_point_with_feedback_root_cause_step(
#         self, mock_continuation_state, mock_event_manager, mock_context
#     ):
#         mock_payload = AutofixRestartFromPointPayload(
#             type=AutofixUpdateType.RESTART_FROM_POINT_WITH_FEEDBACK,
#             step_index=0,
#             retain_insight_card_index=None,
#             message="User feedback",
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_step = MagicMock(spec=DefaultStep)
#         mock_step.key = "root_cause_analysis_processing"
#         mock_step.insights = []
#         mock_state.get.return_value.steps = [mock_step]
#         mock_state.get.return_value.run_id = 123
#         mock_state.update.return_value.__enter__.return_value.steps = [mock_step]
#
#         mock_context.return_value.get_memory.return_value = [
#             Message(content="Initial message", role="system")
#         ]
#
#         restart_from_point_with_feedback(mock_request)
#
#         mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
#         mock_state.update.assert_called()
#         mock_event_manager.return_value.restart_step.assert_called_once_with(mock_step)
#
#         assert len(mock_step.insights) == 1
#         assert isinstance(mock_step.insights[-1], InsightSharingOutput)
#         assert mock_step.insights[-1].insight == "User feedback"
#         assert mock_step.insights[-1].justification == "USER"
#
#     def test_truncate_memory_to_match_insights(self):
#         # Create a mock step with insights
#         mock_step = MagicMock(spec=DefaultStep)
#         mock_step.insights = [
#             MagicMock(spec=InsightSharingOutput, generated_at_memory_index=2),
#             MagicMock(spec=InsightSharingOutput, generated_at_memory_index=4),
#             MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
#         ]
#
#         # Create a mock memory list
#         memory = [
#             Message(content="Initial message", role="system"),
#             Message(content="User message 1", role="user"),
#             Message(content="Assistant response 1", role="assistant"),
#             Message(content="User message 2", role="user"),
#             Message(
#                 content="Assistant response 2",
#                 role="assistant",
#                 tool_calls=[
#                     ToolCall(function="func1", args="{}"),
#                     ToolCall(function="func2", args="{}"),
#                 ],
#             ),
#             Message(content="Tool response 1", role="tool", tool_call_id="func1"),
#             Message(content="Tool response 2", role="tool", tool_call_id="func2"),
#             Message(content="Final message", role="assistant"),
#         ]
#
#         # Test case 1: Normal truncation
#         truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
#         assert len(truncated_memory) == 7  # Up to and including the tool responses
#         assert truncated_memory[-1].role == "tool"
#         assert truncated_memory[-1].content == "Tool response 2"
#
#         # Test case 2: No insights
#         mock_step.insights = []
#         truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
#         assert len(truncated_memory) == 1
#         assert truncated_memory[0].content == "Initial message"
#
#         # Test case 3: All insights have negative index
#         mock_step.insights = [
#             MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
#             MagicMock(spec=InsightSharingOutput, generated_at_memory_index=-1),
#         ]
#         truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
#         assert len(truncated_memory) == len(memory)
#         assert truncated_memory[0].content == "Initial message"
#
#         # Test case 4: Truncation without tool calls
#         mock_step.insights = [MagicMock(spec=InsightSharingOutput, generated_at_memory_index=2)]
#         truncated_memory = truncate_memory_to_match_insights(memory, mock_step)
#         assert len(truncated_memory) == 3
#         assert truncated_memory[-1].content == "Assistant response 1"
#
#         # Test case 5: Truncation with incomplete tool responses
#         mock_step.insights = [MagicMock(spec=InsightSharingOutput, generated_at_memory_index=4)]
#         memory_incomplete = memory[:6]  # Remove the last tool response
#         truncated_memory = truncate_memory_to_match_insights(memory_incomplete, mock_step)
#         assert (
#             len(truncated_memory) == 4
#         )  # Up to Assistant response 2, excluding incomplete tool calls
#         assert truncated_memory[-1].content == "User message 2"
#
#
# class TestUpdateCodeChange:
#     @pytest.fixture
#     def mock_continuation_state(self):
#         with patch("seer.automation.autofix.tasks.ContinuationState") as mock_cs:
#             yield mock_cs
#
#     def test(self):
#         mock_payload = AutofixUserMessagePayload(
#             type=AutofixUpdateType.USER_MESSAGE, text="Invalid payload"
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         with pytest.raises(ValueError, match="Invalid payload type for update_code_change"):
#             update_code_change(mock_request)
#
#     def test_update_code_change_happy_path(self, mock_continuation_state):
#         # Setup
#         mock_payload = AutofixUpdateCodeChangePayload(
#             type=AutofixUpdateType.UPDATE_CODE_CHANGE,
#             repo_id="repo1",
#             hunk_index=0,
#             lines=[
#                 Line(line_type=" ", value="unchanged line"),
#                 Line(line_type="+", value="new line"),
#                 Line(line_type="-", value="removed line"),
#             ],
#             file_path="path/to/file.py",
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_cur_state = MagicMock()
#         mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
#         mock_state.get.return_value = mock_cur_state
#
#         mock_change = MagicMock()
#         mock_change.repo_external_id = "repo1"
#         mock_file_patch = MagicMock(spec=FilePatch)
#         mock_file_patch.path = "path/to/file.py"
#         mock_hunk1 = MagicMock(spec=Hunk)
#         mock_hunk1.target_start = 10
#         mock_hunk1.target_length = 1
#         mock_hunk1.lines = [Line(line_type=" ", value="original line")]
#         mock_hunk2 = MagicMock(spec=Hunk)
#         mock_hunk2.target_start = 15
#         mock_hunk2.target_length = 2
#         mock_hunk2.lines = [Line(line_type=" ", value="another line")]
#         mock_file_patch.hunks = [mock_hunk1, mock_hunk2]
#         mock_change.diff = [mock_file_patch]
#         mock_cur_state.steps[-1].changes = [mock_change]
#
#         # Execute
#         update_code_change(mock_request)
#
#         # Assert
#         mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
#         mock_state.update.assert_called_once()
#
#         # Check if the first hunk was updated correctly
#         updated_hunk1 = mock_file_patch.hunks[0]
#         assert updated_hunk1.lines == mock_payload.lines
#         assert updated_hunk1.target_length == 2  # 1 unchanged + 1 new line
#
#         # Check if the second hunk's start was updated correctly
#         updated_hunk2 = mock_file_patch.hunks[1]
#         assert updated_hunk2.target_start == 16  # 15 + (2 - 1)
#
#         # Verify that the second hunk's length wasn't changed
#         assert updated_hunk2.target_length == 2
#
#     def test_update_code_change_no_matching_change(self, mock_continuation_state):
#         mock_payload = AutofixUpdateCodeChangePayload(
#             type=AutofixUpdateType.UPDATE_CODE_CHANGE,
#             repo_id="non_existent_repo",
#             hunk_index=0,
#             lines=[],
#             file_path="path/to/file.py",
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_cur_state = MagicMock()
#         mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
#         mock_state.get.return_value = mock_cur_state
#         mock_cur_state.steps[-1].changes = []
#
#         with pytest.raises(ValueError, match="No matching change found"):
#             update_code_change(mock_request)
#
#     def test_update_code_change_no_matching_file_patch(self, mock_continuation_state):
#         mock_payload = AutofixUpdateCodeChangePayload(
#             type=AutofixUpdateType.UPDATE_CODE_CHANGE,
#             repo_id="repo1",
#             hunk_index=0,
#             lines=[],
#             file_path="non_existent_file.py",
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_cur_state = MagicMock()
#         mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
#         mock_state.get.return_value = mock_cur_state
#
#         mock_change = MagicMock()
#         mock_change.repo_external_id = "repo1"
#         mock_change.diff = []
#         mock_cur_state.steps[-1].changes = [mock_change]
#
#         with pytest.raises(ValueError, match="No matching file patch found"):
#             update_code_change(mock_request)
#
#     def test_update_code_change_invalid_hunk_index(self, mock_continuation_state):
#         mock_payload = AutofixUpdateCodeChangePayload(
#             type=AutofixUpdateType.UPDATE_CODE_CHANGE,
#             repo_id="repo1",
#             hunk_index=99,  # Invalid index
#             lines=[],
#             file_path="path/to/file.py",
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_cur_state = MagicMock()
#         mock_cur_state.steps = [MagicMock(spec=ChangesStep)]
#         mock_state.get.return_value = mock_cur_state
#
#         mock_change = MagicMock()
#         mock_change.repo_external_id = "repo1"
#         mock_file_patch = MagicMock(spec=FilePatch)
#         mock_file_patch.path = "path/to/file.py"
#         mock_file_patch.hunks = []
#         mock_change.diff = [mock_file_patch]
#         mock_cur_state.steps[-1].changes = [mock_change]
#
#         with pytest.raises(ValueError, match="Hunk index is out of range"):
#             update_code_change(mock_request)
#
#     def test_update_code_change_invalid_step_type(self, mock_continuation_state):
#         mock_payload = AutofixUpdateCodeChangePayload(
#             type=AutofixUpdateType.UPDATE_CODE_CHANGE,
#             repo_id="repo1",
#             hunk_index=0,
#             lines=[],
#             file_path="path/to/file.py",
#         )
#         mock_request = AutofixUpdateRequest(run_id=123, payload=mock_payload)
#
#         mock_state = mock_continuation_state.from_id.return_value
#         mock_cur_state = MagicMock()
#         mock_cur_state.steps = [MagicMock(spec=DefaultStep)]  # Not a ChangesStep
#         mock_state.get.return_value = mock_cur_state
#
#         # Execute
#         update_code_change(mock_request)
#
#         # Assert
#         mock_continuation_state.from_id.assert_called_once_with(123, model=AutofixContinuation)
#         mock_state.update.assert_not_called()  # The function should return early without updating
