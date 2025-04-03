import datetime
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import TimeoutError
from johen import generate

from seer.automation.agent.models import Message, ToolCall
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import InsightSharingType
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixCommentThreadPayload,
    AutofixContinuation,
    AutofixCreatePrUpdatePayload,
    AutofixFeedback,
    AutofixFeedbackPayload,
    AutofixRequest,
    AutofixRequestOptions,
    AutofixResolveCommentThreadPayload,
    AutofixRestartFromPointPayload,
    AutofixRootCauseUpdatePayload,
    AutofixSolutionUpdatePayload,
    AutofixStatus,
    AutofixUpdateCodeChangePayload,
    AutofixUpdateRequest,
    AutofixUpdateType,
    AutofixUserMessagePayload,
    ChangesStep,
    CodebaseChange,
    CommentThread,
    DefaultStep,
    InsightSharingOutput,
    RootCauseStep,
    SolutionStep,
)
from seer.automation.autofix.state import ContinuationState
from seer.automation.autofix.steps.root_cause_step import RootCauseStep as RootCausePipelineStep
from seer.automation.autofix.steps.root_cause_step import RootCauseStepRequest
from seer.automation.autofix.tasks import (
    check_and_mark_recent_autofix_runs,
    comment_on_thread,
    get_autofix_state,
    get_autofix_state_from_pr_id,
    receive_feedback,
    receive_user_message,
    resolve_comment_thread,
    restart_from_point_with_feedback,
    restart_step_with_user_response,
    run_autofix_coding,
    run_autofix_push_changes,
    run_autofix_root_cause,
    truncate_file_changes_to_match_memory,
    truncate_memory_to_match_insights,
    update_code_change,
)
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import FileChange, FilePatch, Hunk, Line, RepoDefinition
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


@pytest.fixture
def autofix_root_cause_and_solution_run():
    with open("tests/data/autofix_root_cause_and_solution_run.json") as f:
        return AutofixContinuation.model_validate_json(f.read())


@pytest.fixture(autouse=True)
def lru_clear():
    yield
    RepoClient.from_repo_definition.cache_clear()


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
@pytest.mark.skip(reason="Flakily causes seg faults.")
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
@pytest.mark.skip(reason="Flaky, needs to be fixed.")
def test_autofix_run_full(autofix_request: AutofixRequest):
    autofix_request.options = AutofixRequestOptions(disable_interactivity=True)
    with eager_celery():
        run_id = run_autofix_root_cause(autofix_request)

    assert run_id is not None

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    root_cause_step = continuation.get().find_step(key="root_cause_analysis")
    assert root_cause_step is not None
    root_cause_step = cast(RootCauseStep, root_cause_step)
    assert root_cause_step.status == AutofixStatus.COMPLETED

    solution_step = continuation.get().find_step(key="solution")
    assert solution_step is not None
    solution_step = cast(SolutionStep, solution_step)
    assert solution_step.status == AutofixStatus.COMPLETED

    with eager_celery():
        run_autofix_coding(
            AutofixUpdateRequest(
                run_id=run_id,
                payload=AutofixSolutionUpdatePayload(custom_solution="fix it however you see fit"),
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
def test_autofix_run_full_without_repos(autofix_request: AutofixRequest):
    autofix_request.options = AutofixRequestOptions(disable_interactivity=True)
    autofix_request.repos = []
    with eager_celery():
        run_id = run_autofix_root_cause(autofix_request)

    assert run_id is not None

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    root_cause_step = continuation.get().find_step(key="root_cause_analysis")
    assert root_cause_step is not None
    root_cause_step = cast(RootCauseStep, root_cause_step)
    assert root_cause_step.status == AutofixStatus.COMPLETED

    solution_step = continuation.get().find_step(key="solution")
    assert solution_step is not None
    solution_step = cast(SolutionStep, solution_step)
    assert solution_step.status == AutofixStatus.COMPLETED

    assert continuation.get().status not in {AutofixStatus.ERROR}

    with eager_celery():
        with pytest.raises(ValueError):
            run_autofix_coding(
                AutofixUpdateRequest(
                    run_id=run_id,
                    payload=AutofixSolutionUpdatePayload(
                        custom_solution="fix it however you see fit"
                    ),
                )
            )

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    changes_step = continuation.get().find_step(key="changes")

    assert changes_step is None

    assert continuation.get().status not in {AutofixStatus.ERROR}  # Shouldn't error out the run


@pytest.mark.vcr()
@pytest.mark.skip(reason="Flaky, needs to be fixed.")
def test_autofix_run_full_with_partial_supported_repos(autofix_request: AutofixRequest):
    autofix_request.options = AutofixRequestOptions(disable_interactivity=True)
    autofix_request.repos = [
        RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="439438299"),
        RepoDefinition(
            provider="bitbucket", owner="getsentry", name="wowowoww", external_id="2344"
        ),
    ]
    with eager_celery():
        run_id = run_autofix_root_cause(autofix_request)

    assert run_id is not None

    continuation = get_autofix_state(run_id=run_id)

    assert continuation is not None

    root_cause_step = continuation.get().find_step(key="root_cause_analysis")
    assert root_cause_step is not None
    root_cause_step = cast(RootCauseStep, root_cause_step)
    assert root_cause_step.status == AutofixStatus.COMPLETED

    solution_step = continuation.get().find_step(key="solution")
    assert solution_step is not None
    solution_step = cast(SolutionStep, solution_step)
    assert solution_step.status == AutofixStatus.COMPLETED

    assert continuation.get().status not in {AutofixStatus.ERROR}

    with eager_celery():
        run_autofix_coding(
            AutofixUpdateRequest(
                run_id=run_id,
                payload=AutofixSolutionUpdatePayload(custom_solution="fix it however you see fit"),
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
@pytest.mark.skip(reason="Flakily causes seg faults.")
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
@pytest.mark.skip(reason="Flaky")
def test_autofix_run_coding(autofix_root_cause_and_solution_run: AutofixContinuation):
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_root_cause_and_solution_run.run_id,
                group_id=1,
                value=autofix_root_cause_and_solution_run.model_dump(mode="json"),
            )
        )
        session.commit()

    with eager_celery():
        run_autofix_coding(
            AutofixUpdateRequest(
                run_id=autofix_root_cause_and_solution_run.run_id,
                payload=AutofixSolutionUpdatePayload(solution_selected=True, mode="fix"),
            )
        )

    continuation = get_autofix_state(run_id=autofix_root_cause_and_solution_run.run_id)

    assert continuation is not None

    changes_step = continuation.get().find_step(key="changes")

    assert changes_step is not None
    changes_step = cast(ChangesStep, changes_step)
    assert changes_step.status == AutofixStatus.COMPLETED
    assert len(changes_step.changes) > 0

    assert continuation.get().status not in {AutofixStatus.ERROR}


@pytest.mark.vcr()
@pytest.mark.skip(reason="Flakily causes seg faults.")
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


@pytest.mark.skip(reason="Failing in prod even when re-creating cassettes.")
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


@pytest.mark.vcr()
def test_receive_user_message_response_to_question(autofix_root_cause_run: AutofixContinuation):
    # Create initial state with a step waiting for response
    autofix_root_cause_run.steps = autofix_root_cause_run.steps[:1]
    step = autofix_root_cause_run.steps[0]
    step.status = AutofixStatus.WAITING_FOR_USER_RESPONSE
    step.insights = []

    # Store initial state in database
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_root_cause_run.run_id,
                group_id=100,
                value=autofix_root_cause_run.model_dump(mode="json"),
            )
        )
        session.commit()

    # Create context with memory
    context = AutofixContext.from_run_id(autofix_root_cause_run.run_id)
    context.store_memory(
        "root_cause_analysis",
        [
            Message(content="Test Prompt", role="user"),
            Message(
                content="Test Question",
                tool_calls=[
                    ToolCall(function="func", args='{"question": "Test Question"}', id="tool_1")
                ],
                role="assistant",
            ),
        ],
    )

    # Create and send user message request
    payload = AutofixUserMessagePayload(
        type=AutofixUpdateType.USER_MESSAGE, text="This is the user response"
    )
    request = AutofixUpdateRequest(run_id=autofix_root_cause_run.run_id, payload=payload)

    with eager_celery():
        receive_user_message(request)

    # Verify state was updated correctly
    updated_state = get_autofix_state(run_id=autofix_root_cause_run.run_id)
    assert updated_state is not None
    updated_step = updated_state.get().steps[-1]

    assert isinstance(updated_step, DefaultStep)
    assert len(updated_step.insights) > 0
    assert "This is the user response" in updated_step.insights[-1].insight


def test_receive_user_message_interjection(autofix_root_cause_run: AutofixContinuation):
    # Create initial state with a step not waiting for response
    autofix_root_cause_run.steps = autofix_root_cause_run.steps[:1]
    step = autofix_root_cause_run.steps[0]
    step.status = AutofixStatus.PROCESSING
    step.insights = []

    # Store initial state in database
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_root_cause_run.run_id,
                group_id=200,
                value=autofix_root_cause_run.model_dump(mode="json"),
            )
        )
        session.commit()

    # Create and send user message request
    payload = AutofixUserMessagePayload(
        type=AutofixUpdateType.USER_MESSAGE, text="User interjection"
    )
    request = AutofixUpdateRequest(run_id=autofix_root_cause_run.run_id, payload=payload)

    with eager_celery():
        receive_user_message(request)

    # Verify state was updated correctly
    updated_state = get_autofix_state(run_id=autofix_root_cause_run.run_id)
    assert updated_state is not None
    updated_step = updated_state.get().steps[-1]
    assert isinstance(updated_step, DefaultStep)
    assert len(updated_step.insights) > 0
    assert updated_step.insights[-1].insight == "User interjection"
    assert updated_step.insights[-1].justification == "USER"


def test_receive_user_message_invalid_payload_type(autofix_root_cause_run: AutofixContinuation):
    payload = AutofixRootCauseUpdatePayload(
        type=AutofixUpdateType.SELECT_ROOT_CAUSE,
        cause_id=None,
        custom_root_cause="Invalid payload type",
    )
    request = AutofixUpdateRequest(run_id=autofix_root_cause_run.run_id, payload=payload)

    with pytest.raises(ValueError, match="Invalid payload type for user_message"):
        with eager_celery():
            receive_user_message(request)


@pytest.mark.vcr()
def test_restart_step_with_user_response(autofix_root_cause_run: AutofixContinuation):
    # Create initial state with a step
    autofix_root_cause_run.steps = autofix_root_cause_run.steps[:1]
    step = autofix_root_cause_run.steps[0]
    step.status = AutofixStatus.WAITING_FOR_USER_RESPONSE
    step.insights = []

    # Store initial state in database
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_root_cause_run.run_id,
                group_id=300,
                value=autofix_root_cause_run.model_dump(mode="json"),
            )
        )
        session.commit()

    # Create context with memory
    context = AutofixContext.from_run_id(autofix_root_cause_run.run_id)
    memory = [Message(role="assistant", content="Previous message", tool_call_id="tool_1")]
    context.store_memory("root_cause_analysis", memory)

    # Create event manager
    event_manager = AutofixEventManager(ContinuationState(autofix_root_cause_run.run_id))

    with eager_celery():
        restart_step_with_user_response(
            ContinuationState(autofix_root_cause_run.run_id),
            memory,
            "User response",
            event_manager,
            step,
            RootCausePipelineStep,
            RootCauseStepRequest,
        )

    # Verify memory was updated correctly
    assert len(memory) == 2
    assert memory[-1].role == "tool"
    assert memory[-1].content == "User response"
    assert memory[-1].tool_call_id == "tool_1"


def test_truncate_memory_to_match_insights():
    # Create a step with insights
    step = DefaultStep(
        title="Test Step",
        status=AutofixStatus.COMPLETED,
        initial_memory_length=1,
        insights=[
            InsightSharingOutput(
                insight="Insight 1",
                justification="TEST",
                codebase_context=[],
                stacktrace_context=[],
                breadcrumb_context=[],
                generated_at_memory_index=2,
            ),
            InsightSharingOutput(
                insight="Insight 2",
                justification="TEST",
                codebase_context=[],
                stacktrace_context=[],
                breadcrumb_context=[],
                generated_at_memory_index=4,
                type=InsightSharingType.FILE_CHANGE,
            ),
            InsightSharingOutput(
                insight="Insight 3",
                justification="TEST",
                codebase_context=[],
                stacktrace_context=[],
                breadcrumb_context=[],
                generated_at_memory_index=-1,
            ),
        ],
    )

    # Create a memory list
    memory = [
        Message(content="Initial message", role="user"),
        Message(content="Assistant response 1", role="assistant"),
        Message(content="User message 2", role="user"),
        Message(
            content="Assistant response 2",
            role="assistant",
            tool_calls=[
                ToolCall(function="func1", args="{}", id="tool_1"),
                ToolCall(function="func2", args="{}", id="tool_2"),
            ],
        ),
        Message(content="Tool response 1", role="tool", tool_call_id="tool_1"),
        Message(content="Tool response 2", role="tool", tool_call_id="tool_2"),
        Message(content="Final message", role="assistant"),
    ]

    # Normal truncation with instructions and tool calls
    truncated_memory = truncate_memory_to_match_insights(memory, 3, step, False, True)
    assert len(truncated_memory) == 6  # Up to and including tool responses
    assert truncated_memory[-2].content == "Tool response 1"
    assert truncated_memory[-1].content == "Tool response 2"

    # Normal truncation without instructions and tool calls
    truncated_memory = truncate_memory_to_match_insights(memory, 3, step, False, False)
    assert len(truncated_memory) == 3  # Up to and including tool responses

    # No insights
    step.insights = []
    truncated_memory = truncate_memory_to_match_insights(memory, 0, step, False, False)
    assert (
        len(truncated_memory) == step.initial_memory_length
    )  # Should use initial_memory_length when no insights
    assert truncated_memory[-1].content == "Initial message"

    # Truncation without tool calls
    step.insights = [
        InsightSharingOutput(
            insight="Insight 1",
            justification="TEST",
            codebase_context=[],
            stacktrace_context=[],
            breadcrumb_context=[],
            generated_at_memory_index=2,
        )
    ]
    truncated_memory = truncate_memory_to_match_insights(memory, 2, step, False, False)
    assert len(truncated_memory) == 3
    assert truncated_memory[-1].content == "User message 2"

    # Truncation with incomplete tool responses
    step.insights = [
        InsightSharingOutput(
            insight="Insight 1",
            justification="TEST",
            codebase_context=[],
            stacktrace_context=[],
            breadcrumb_context=[],
            generated_at_memory_index=4,
        )
    ]
    memory_incomplete = memory[:4]  # Remove the last tool response
    truncated_memory = truncate_memory_to_match_insights(memory_incomplete, 4, step, False, False)
    assert len(truncated_memory) == 4  # Should exclude incomplete tool calls
    assert truncated_memory[-1].content == "Assistant response 2"

    # Memory index less than initial_memory_length
    step.initial_memory_length = 3
    step.insights = []
    truncated_memory = truncate_memory_to_match_insights(memory, 1, step, False, False)
    assert len(truncated_memory) == 3
    assert truncated_memory[-1].content == "User message 2"

    # Memory index beyond memory length
    truncated_memory = truncate_memory_to_match_insights(
        memory, len(memory) + 1, step, False, False
    )
    assert len(truncated_memory) == len(memory)  # Should return full memory
    assert truncated_memory[-1].content == "Final message"

    # Memory index is -1
    truncated_memory = truncate_memory_to_match_insights(memory, -1, step, False, False)
    assert len(truncated_memory) == len(memory)  # Should return full memory
    assert truncated_memory[-1].content == "Final message"

    # Edit insight with tool call but no rethink instruction
    step.insights = [
        InsightSharingOutput(
            insight="Edit Insight",
            justification="TEST",
            codebase_context=[],
            stacktrace_context=[],
            breadcrumb_context=[],
            generated_at_memory_index=3,
            type=InsightSharingType.FILE_CHANGE,
        )
    ]
    truncated_memory = truncate_memory_to_match_insights(memory, 3, step, True, False)
    assert len(truncated_memory) == 3  # Should exclude the tool call message for edit insight
    assert truncated_memory[-1].content == "User message 2"

    # Edit insight with tool call and rethink instruction
    truncated_memory = truncate_memory_to_match_insights(memory, 3, step, True, True)
    assert len(truncated_memory) == 6
    assert (
        truncated_memory[4].content
        == "Notice: This tool call did not apply, the user provided instructions."
    )
    assert (
        truncated_memory[5].content
        == "Notice: This tool call did not apply, the user provided instructions."
    )


def test_truncate_file_changes_to_match_memory():
    memory = [
        Message(content="Initial message", role="user"),
        Message(
            content="Assistant response with tools",
            role="assistant",
            tool_calls=[
                ToolCall(function="edit_file", args="{}", id="tool_1"),
                ToolCall(function="rename_file", args="{}", id="tool_2"),
            ],
        ),
        Message(content="Tool response 1", role="tool", tool_call_id="tool_1"),
        Message(content="Tool response 2", role="tool", tool_call_id="tool_2"),
        Message(
            content="Second assistant response",
            role="assistant",
            tool_calls=[
                ToolCall(function="create_file", args="{}", id="tool_3"),
            ],
        ),
        Message(content="Tool response 3", role="tool", tool_call_id="tool_3"),
    ]

    # Create file changes
    file_changes = [
        FileChange(
            path="file1.py",
            tool_call_id="tool_1",
            change_type="edit",
            reference_snippet="content1",
            new_snippet="content1_new",
        ),
        FileChange(
            path="file2.py", tool_call_id="tool_2", change_type="create", new_snippet="content2"
        ),
        FileChange(
            path="file3.py",
            tool_call_id="tool_3",
            change_type="delete",
            reference_snippet="content3",
        ),
        FileChange(
            path="file4.py",
            tool_call_id="tool_4",
            change_type="edit",
            reference_snippet="content4",
            new_snippet="content4_new",
        ),
    ]

    # Test case 1: All memory present
    truncated_changes = truncate_file_changes_to_match_memory(file_changes, memory)
    assert len(truncated_changes) == 3
    assert set(fc.path for fc in truncated_changes) == {"file1.py", "file2.py", "file3.py"}

    # Test case 2: Truncated memory
    truncated_memory = memory[:4]  # Only includes tool_1 and tool_2
    truncated_changes = truncate_file_changes_to_match_memory(file_changes, truncated_memory)
    assert len(truncated_changes) == 2
    assert set(fc.path for fc in truncated_changes) == {"file1.py", "file2.py"}

    # Test case 3: Empty memory
    truncated_changes = truncate_file_changes_to_match_memory(file_changes, [])
    assert len(truncated_changes) == 0

    # Test case 4: Memory with no tool calls
    memory_no_tools = [
        Message(content="Initial message", role="user"),
        Message(content="Assistant response", role="assistant"),
    ]
    truncated_changes = truncate_file_changes_to_match_memory(file_changes, memory_no_tools)
    assert len(truncated_changes) == 0


def test_update_code_change_happy_path():
    # Create initial state with a changes step
    state = next(generate(AutofixContinuation))
    file_patch = FilePatch(
        type="A",
        added=0,
        removed=0,
        source_file="",
        target_file="",
        path="test_file.py",
        hunks=[
            Hunk(
                source_start=0,
                source_length=0,
                section_header="",
                target_start=10,
                target_length=3,
                lines=[
                    Line(line_type=" ", value="def test():", target_line_no=10),
                    Line(line_type="-", value="    return False", target_line_no=11),
                    Line(line_type=" ", value="    pass", target_line_no=12),
                ],
            ),
            Hunk(
                source_start=0,
                source_length=0,
                section_header="",
                target_start=20,
                target_length=2,
                lines=[
                    Line(line_type=" ", value="def another_test():", target_line_no=20),
                    Line(line_type=" ", value="    pass", target_line_no=21),
                ],
            ),
        ],
    )
    changes_step = ChangesStep(
        title="Test Change",
        status=AutofixStatus.COMPLETED,
        changes=[
            CodebaseChange(
                repo_external_id="test_repo",
                repo_name="test/repo",
                title="Test Change",
                description="Test Description",
                diff=[file_patch],
            )
        ],
    )
    state.steps = [changes_step]

    # Store state in database
    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    # Create update request
    new_lines = [
        Line(line_type=" ", value="def test():", target_line_no=10),
        Line(line_type="+", value="    return True", target_line_no=11),
        Line(line_type=" ", value="    pass", target_line_no=12),
    ]
    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_external_id="test_repo",
            hunk_index=0,
            lines=new_lines,
            file_path="test_file.py",
        ),
    )

    # Execute update
    update_code_change(request)

    # Verify changes
    updated_state = get_autofix_state(run_id=1)
    assert updated_state is not None
    changes_step = updated_state.get().steps[-1]
    assert isinstance(changes_step, ChangesStep)
    updated_changes = changes_step.changes[0]
    updated_hunk = updated_changes.diff[0].hunks[0]

    # Check updated hunk lines
    assert len(updated_hunk.lines) == 3
    assert updated_hunk.lines[0].value == "def test():"
    assert updated_hunk.lines[1].value == "    return True"
    assert updated_hunk.lines[2].value == "    pass"

    # Check line numbers were updated correctly
    assert updated_hunk.lines[0].target_line_no == 10
    assert updated_hunk.lines[1].target_line_no == 11
    assert updated_hunk.lines[2].target_line_no == 12

    # Check subsequent hunk was not affected
    subsequent_hunk = updated_changes.diff[0].hunks[1]
    assert subsequent_hunk.target_start == 20  # Should remain unchanged
    assert subsequent_hunk.target_length == 2


def test_update_code_change_happy_path_with_old_repo_id():
    # Create initial state with a changes step
    state = next(generate(AutofixContinuation))
    file_patch = FilePatch(
        type="A",
        added=0,
        removed=0,
        source_file="",
        target_file="",
        path="test_file.py",
        hunks=[
            Hunk(
                source_start=0,
                source_length=0,
                section_header="",
                target_start=10,
                target_length=3,
                lines=[
                    Line(line_type=" ", value="def test():", target_line_no=10),
                    Line(line_type="-", value="    return False", target_line_no=11),
                    Line(line_type=" ", value="    pass", target_line_no=12),
                ],
            ),
            Hunk(
                source_start=0,
                source_length=0,
                section_header="",
                target_start=20,
                target_length=2,
                lines=[
                    Line(line_type=" ", value="def another_test():", target_line_no=20),
                    Line(line_type=" ", value="    pass", target_line_no=21),
                ],
            ),
        ],
    )
    changes_step = ChangesStep(
        title="Test Change",
        status=AutofixStatus.COMPLETED,
        changes=[
            CodebaseChange(
                repo_external_id="test_repo",
                repo_name="test/repo",
                title="Test Change",
                description="Test Description",
                diff=[file_patch],
            )
        ],
    )
    state.steps = [changes_step]

    # Store state in database
    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    # Create update request
    new_lines = [
        Line(line_type=" ", value="def test():", target_line_no=10),
        Line(line_type="+", value="    return True", target_line_no=11),
        Line(line_type=" ", value="    pass", target_line_no=12),
    ]
    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_id="test_repo",
            hunk_index=0,
            lines=new_lines,
            file_path="test_file.py",
        ),
    )

    # Execute update
    update_code_change(request)

    # Verify changes
    updated_state = get_autofix_state(run_id=1)
    assert updated_state is not None
    changes_step = updated_state.get().steps[-1]
    assert isinstance(changes_step, ChangesStep)
    updated_changes = changes_step.changes[0]
    updated_hunk = updated_changes.diff[0].hunks[0]

    # Check that the updated applied
    assert len(updated_hunk.lines) == 3


def test_update_code_change_invalid_payload_type():
    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUserMessagePayload(
            type=AutofixUpdateType.USER_MESSAGE,
            text="Invalid payload",
        ),
    )

    with pytest.raises(ValueError, match="Invalid payload type for update_code_change"):
        update_code_change(request)


def test_update_code_change_no_matching_repo():
    # Create initial state with a changes step but different repo ID
    state = next(generate(AutofixContinuation))
    changes_step = ChangesStep(
        title="Test Change",
        status=AutofixStatus.COMPLETED,
        changes=[
            CodebaseChange(
                repo_external_id="different_repo",
                repo_name="test/repo",
                title="Test Change",
                description="Test Description",
                diff=[],
            )
        ],
    )
    state.steps = [changes_step]

    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_external_id="test_repo",
            hunk_index=0,
            lines=[],
            file_path="test_file.py",
        ),
    )

    with pytest.raises(ValueError, match="No matching change found"):
        update_code_change(request)


def test_update_code_change_no_matching_file():
    # Create initial state with a changes step but no matching file
    state = next(generate(AutofixContinuation))
    changes_step = ChangesStep(
        title="Test Change",
        status=AutofixStatus.COMPLETED,
        changes=[
            CodebaseChange(
                repo_external_id="test_repo",
                repo_name="test/repo",
                title="Test Change",
                description="Test Description",
                diff=[
                    FilePatch(
                        type="A",
                        added=0,
                        removed=0,
                        source_file="",
                        target_file="",
                        path="different_file.py",
                        hunks=[],
                    ),
                ],
            )
        ],
    )
    state.steps = [changes_step]

    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_external_id="test_repo",
            hunk_index=0,
            lines=[],
            file_path="test_file.py",
        ),
    )

    with pytest.raises(ValueError, match="No matching file patch found"):
        update_code_change(request)


def test_update_code_change_invalid_hunk_index():
    # Create initial state with a changes step but invalid hunk index
    state = next(generate(AutofixContinuation))
    file_patch = FilePatch(
        type="A",
        added=0,
        removed=0,
        source_file="",
        target_file="",
        path="test_file.py",
        hunks=[
            Hunk(
                source_start=0,
                source_length=0,
                section_header="",
                target_start=10,
                target_length=1,
                lines=[Line(line_type=" ", value="def test():", target_line_no=10)],
            ),
        ],
    )
    changes_step = ChangesStep(
        title="Test Change",
        status=AutofixStatus.COMPLETED,
        changes=[
            CodebaseChange(
                repo_external_id="test_repo",
                repo_name="test/repo",
                title="Test Change",
                description="Test Description",
                diff=[file_patch],
            )
        ],
    )
    state.steps = [changes_step]

    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_external_id="test_repo",
            hunk_index=99,  # Invalid index
            lines=[],
            file_path="test_file.py",
        ),
    )

    with pytest.raises(ValueError, match="Hunk index is out of range"):
        update_code_change(request)


def test_update_code_change_non_changes_step():
    # Create initial state with a non-changes step
    state = next(generate(AutofixContinuation))
    state.steps = [DefaultStep(title="Test Step", status=AutofixStatus.COMPLETED)]

    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUpdateCodeChangePayload(
            type=AutofixUpdateType.UPDATE_CODE_CHANGE,
            repo_external_id="test_repo",
            hunk_index=0,
            lines=[],
            file_path="test_file.py",
        ),
    )

    # Should return silently without making changes
    update_code_change(request)

    # Verify state wasn't modified
    updated_state = get_autofix_state(run_id=1)
    assert updated_state is not None
    assert isinstance(updated_state.get().steps[-1], DefaultStep)


@pytest.mark.vcr()
def test_comment_on_thread_creates_new_thread():
    # Create initial state with a step
    state = next(generate(AutofixContinuation))
    step = DefaultStep(title="Test Step", status=AutofixStatus.COMPLETED)
    state.steps = [step]

    # Store state in database
    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    # Create request with new thread
    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixCommentThreadPayload(
            type=AutofixUpdateType.COMMENT_THREAD,
            thread_id="new_thread",
            step_index=0,
            message="Test comment",
            selected_text="Test selection",
        ),
    )

    comment_on_thread(request)

    # Verify thread was created
    updated_state = get_autofix_state(run_id=1)
    assert updated_state is not None
    updated_step = updated_state.get().steps[0]
    assert updated_step.active_comment_thread is not None
    assert updated_step.active_comment_thread.id == "new_thread"
    assert updated_step.active_comment_thread.selected_text == "Test selection"
    assert len(updated_step.active_comment_thread.messages) == 2
    assert updated_step.active_comment_thread.messages[0].content == "Test comment"
    # Don't assert exact response content since it may vary
    assert updated_step.active_comment_thread.messages[1].role == "assistant"
    assert updated_step.active_comment_thread.messages[1].content


@pytest.mark.vcr()
def test_comment_on_thread_updates_existing_thread():
    # Create initial state with a step that has an existing thread
    state = next(generate(AutofixContinuation))
    step = DefaultStep(
        title="Test Step",
        status=AutofixStatus.COMPLETED,
        active_comment_thread=CommentThread(
            id="existing_thread",
            selected_text="Original selection",
            messages=[
                Message(role="user", content="Previous comment"),
                Message(role="assistant", content="Previous response"),
            ],
        ),
    )
    state.steps = [step]

    # Store state in database
    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    # Create request for existing thread
    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixCommentThreadPayload(
            type=AutofixUpdateType.COMMENT_THREAD,
            thread_id="existing_thread",
            step_index=0,
            message="New comment",
            selected_text="Updated selection",
        ),
    )

    comment_on_thread(request)

    # Verify thread was updated
    updated_state = get_autofix_state(run_id=1)
    assert updated_state is not None
    updated_step = updated_state.get().steps[0]
    assert updated_step.active_comment_thread is not None
    assert updated_step.active_comment_thread.id == "existing_thread"
    assert updated_step.active_comment_thread.selected_text == "Updated selection"
    assert len(updated_step.active_comment_thread.messages) == 4
    assert updated_step.active_comment_thread.messages[-2].content == "New comment"
    assert updated_step.active_comment_thread.messages[-1].role == "assistant"
    assert updated_step.active_comment_thread.messages[-1].content


@pytest.mark.vcr()
@pytest.mark.skip(reason="Flakily causes seg faults.")
def test_comment_on_thread_with_action_requested(autofix_root_cause_run: AutofixContinuation):
    # Set up initial state with a step that has a comment thread
    autofix_root_cause_run.steps = autofix_root_cause_run.steps[:1]  # Keep only first step
    step = autofix_root_cause_run.steps[0]
    step.active_comment_thread = CommentThread(
        id="thread_with_action",
        selected_text="Test selection",
        messages=[],
    )
    step.insights = []

    # Store state in database
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_root_cause_run.run_id,
                group_id=1,
                value=autofix_root_cause_run.model_dump(mode="json"),
            )
        )
        session.commit()

    # Set up memory context
    context = AutofixContext.from_run_id(autofix_root_cause_run.run_id)
    context.store_memory(
        "root_cause_analysis",
        [
            Message(content="Initial analysis", role="user"),
            Message(content="Initial response", role="assistant"),
        ],
    )

    # Create request
    request = AutofixUpdateRequest(
        run_id=autofix_root_cause_run.run_id,
        payload=AutofixCommentThreadPayload(
            type=AutofixUpdateType.COMMENT_THREAD,
            thread_id="thread_with_action",
            step_index=0,
            message="Please revise the analysis based on this feedback. Action is needed.",
            selected_text="Test selection",
            retain_insight_card_index=-1,
        ),
    )

    with eager_celery():
        comment_on_thread(request)

    # Verify thread was marked as completed
    updated_state = get_autofix_state(run_id=autofix_root_cause_run.run_id)
    assert updated_state is not None
    updated_step = updated_state.get().steps[0]
    assert updated_step.active_comment_thread is not None
    assert updated_step.active_comment_thread.is_completed is True

    # Verify memory was updated correctly
    new_run_memory = context.get_memory("root_cause_analysis")
    assert len(new_run_memory) > 2
    assert "Please revise" in new_run_memory[-2].content
    assert new_run_memory[-1].role == "assistant"


def test_comment_on_thread_invalid_payload():
    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixUserMessagePayload(
            type=AutofixUpdateType.USER_MESSAGE,
            text="Invalid payload",
        ),
    )

    with pytest.raises(ValueError, match="Invalid payload type for comment_on_thread"):
        comment_on_thread(request)


def test_resolve_comment_thread():
    # Create initial state with a step that has an existing thread
    state = next(generate(AutofixContinuation))
    step = DefaultStep(
        title="Test Step",
        status=AutofixStatus.COMPLETED,
        active_comment_thread=CommentThread(
            id="existing_thread",
            selected_text="Test selection",
            messages=[
                Message(role="user", content="Test comment"),
                Message(role="assistant", content="Test response"),
            ],
        ),
    )
    state.steps = [step]

    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixResolveCommentThreadPayload(
            type=AutofixUpdateType.RESOLVE_COMMENT_THREAD,
            thread_id="existing_thread",
            step_index=0,
            is_agent_comment=False,
        ),
    )

    resolve_comment_thread(request)

    # Verify thread was removed
    updated_state = get_autofix_state(run_id=1)
    assert updated_state is not None
    updated_step = updated_state.get().steps[0]
    assert updated_step.active_comment_thread is None


def test_resolve_agent_comment_thread():
    # Create initial state with two steps, where the second step has an agent comment thread
    state = next(generate(AutofixContinuation))
    step1 = DefaultStep(
        title="First Step",
        status=AutofixStatus.COMPLETED,
    )
    step2 = DefaultStep(
        title="Second Step",
        status=AutofixStatus.COMPLETED,
        agent_comment_thread=CommentThread(
            id="agent_thread",
            selected_text="Agent selection",
            messages=[
                Message(role="assistant", content="Agent comment"),
                Message(role="user", content="User response"),
            ],
        ),
    )
    state.steps = [step1, step2]

    with Session() as session:
        session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    request = AutofixUpdateRequest(
        run_id=1,
        payload=AutofixResolveCommentThreadPayload(
            type=AutofixUpdateType.RESOLVE_COMMENT_THREAD,
            thread_id="agent_thread",
            step_index=0,  # The index of the processing step, not where the thread is stored
            is_agent_comment=True,
        ),
    )

    resolve_comment_thread(request)

    # Verify thread was removed
    updated_state = get_autofix_state(run_id=1)
    assert updated_state is not None
    updated_step2 = updated_state.get().steps[1]
    assert updated_step2.agent_comment_thread is None


@pytest.fixture
def autofix_state_for_feedback():
    """Create a test autofix state with no feedback."""
    state = next(generate(AutofixContinuation))
    state.feedback = None

    with Session() as session:
        session.add(DbRunState(id=100, group_id=100, value=state.model_dump(mode="json")))
        session.commit()

    return 100  # Return the run_id


@pytest.fixture
def autofix_state_with_feedback():
    """Create a test autofix state with existing feedback."""
    state = next(generate(AutofixContinuation))
    state.feedback = AutofixFeedback(
        root_cause_thumbs_up=True,
        root_cause_thumbs_down=False,
        solution_thumbs_up=None,
        solution_thumbs_down=None,
    )

    with Session() as session:
        session.add(DbRunState(id=101, group_id=101, value=state.model_dump(mode="json")))
        session.commit()

    return 101  # Return the run_id


def test_receive_feedback_root_cause_thumbs_up(autofix_state_for_feedback):
    """Test receiving root cause thumbs up feedback."""
    request = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="root_cause_thumbs_up"
        ),
    )

    receive_feedback(request)

    # Verify the state was updated correctly
    state = ContinuationState(autofix_state_for_feedback)
    cur = state.get()

    assert cur.feedback is not None
    assert cur.feedback.root_cause_thumbs_up is True
    assert cur.feedback.root_cause_thumbs_down is False
    assert cur.feedback.solution_thumbs_up is None
    assert cur.feedback.solution_thumbs_down is None


def test_receive_feedback_root_cause_thumbs_down(autofix_state_for_feedback):
    """Test receiving root cause thumbs down feedback."""
    request = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="root_cause_thumbs_down"
        ),
    )

    receive_feedback(request)

    # Verify the state was updated correctly
    state = ContinuationState(autofix_state_for_feedback)
    cur = state.get()

    assert cur.feedback is not None
    assert cur.feedback.root_cause_thumbs_up is False
    assert cur.feedback.root_cause_thumbs_down is True
    assert cur.feedback.solution_thumbs_up is None
    assert cur.feedback.solution_thumbs_down is None


def test_receive_feedback_solution_thumbs_up(autofix_state_for_feedback):
    """Test receiving solution thumbs up feedback."""
    request = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="solution_thumbs_up"
        ),
    )

    receive_feedback(request)

    # Verify the state was updated correctly
    state = ContinuationState(autofix_state_for_feedback)
    cur = state.get()

    assert cur.feedback is not None
    assert cur.feedback.root_cause_thumbs_up is None
    assert cur.feedback.root_cause_thumbs_down is None
    assert cur.feedback.solution_thumbs_up is True
    assert cur.feedback.solution_thumbs_down is False


def test_receive_feedback_solution_thumbs_down(autofix_state_for_feedback):
    """Test receiving solution thumbs down feedback."""
    request = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="solution_thumbs_down"
        ),
    )

    receive_feedback(request)

    # Verify the state was updated correctly
    state = ContinuationState(autofix_state_for_feedback)
    cur = state.get()

    assert cur.feedback is not None
    assert cur.feedback.root_cause_thumbs_up is None
    assert cur.feedback.root_cause_thumbs_down is None
    assert cur.feedback.solution_thumbs_up is False
    assert cur.feedback.solution_thumbs_down is True


def test_receive_feedback_existing_feedback(autofix_state_for_feedback):
    """Test receiving feedback when feedback already exists."""
    # First set some initial feedback
    request1 = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="root_cause_thumbs_up"
        ),
    )
    receive_feedback(request1)

    # Then update with new feedback
    request2 = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="solution_thumbs_up"
        ),
    )
    receive_feedback(request2)

    # Verify both feedback items were saved
    state = ContinuationState(autofix_state_for_feedback)
    cur = state.get()

    assert cur.feedback is not None
    assert cur.feedback.root_cause_thumbs_up is True
    assert cur.feedback.root_cause_thumbs_down is False
    assert cur.feedback.solution_thumbs_up is True
    assert cur.feedback.solution_thumbs_down is False


def test_receive_feedback_toggle(autofix_state_for_feedback):
    """Test toggling feedback from thumbs up to thumbs down."""
    # First set thumbs up
    request1 = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="root_cause_thumbs_up"
        ),
    )
    receive_feedback(request1)

    # Then change to thumbs down
    request2 = AutofixUpdateRequest(
        run_id=autofix_state_for_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="root_cause_thumbs_down"
        ),
    )
    receive_feedback(request2)

    # Verify feedback was toggled
    state = ContinuationState(autofix_state_for_feedback)
    cur = state.get()

    assert cur.feedback is not None
    assert cur.feedback.root_cause_thumbs_up is False
    assert cur.feedback.root_cause_thumbs_down is True


def test_receive_feedback_update_existing_feedback(autofix_state_with_feedback):
    """Test receiving feedback when feedback object already exists."""
    request = AutofixUpdateRequest(
        run_id=autofix_state_with_feedback,
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="solution_thumbs_up"
        ),
    )

    receive_feedback(request)

    # Verify the state was updated correctly
    state = ContinuationState(autofix_state_with_feedback)
    cur = state.get()

    assert cur.feedback is not None
    assert cur.feedback.root_cause_thumbs_up is True
    assert cur.feedback.root_cause_thumbs_down is False
    assert cur.feedback.solution_thumbs_up is True
    assert cur.feedback.solution_thumbs_down is False


def test_receive_feedback_invalid_run_id():
    """Test receiving feedback for a non-existent run."""
    request = AutofixUpdateRequest(
        run_id=999999,  # Non-existent run ID
        payload=AutofixFeedbackPayload(
            type=AutofixUpdateType.FEEDBACK, action="root_cause_thumbs_up"
        ),
    )

    with pytest.raises(ValueError, match="Autofix state not found"):
        receive_feedback(request)


@pytest.mark.parametrize(
    "scenario, mock_get_side_effect, expected_status, expected_message",
    [
        (
            "success",
            [MagicMock()],  # Successful result
            "success",
            None,
        ),
        (
            "retry_success",
            [ConnectionError("Temporary hiccup"), MagicMock()],  # Connection error then success
            "success",
            None,
        ),
        (
            "wait_timeout",
            TimeoutError("Wait timed out"),  # task times out
            "error",
            "GitHub didn't respond - maybe try again?",
        ),
    ],
)
@patch("seer.automation.autofix.tasks.commit_changes_task.apply_async")
@patch(
    "seer.automation.autofix.tasks.COMMIT_CHANGE_TASK_TIMEOUT", 0.1
)  # Patch timeout for faster test
def test_run_autofix_push_changes_with_retry(
    mock_apply_async,
    scenario,
    mock_get_side_effect,
    expected_status,
    expected_message,
    autofix_full_finished_run: AutofixContinuation,
):
    # Setup DB state
    with Session() as session:
        session.add(
            DbRunState(
                id=autofix_full_finished_run.run_id,
                group_id=1,
                value=autofix_full_finished_run.model_dump(mode="json"),
            )
        )
        session.commit()

    # Configure mock AsyncResult
    mock_async_result = MagicMock()
    mock_async_result.get.side_effect = mock_get_side_effect
    mock_apply_async.return_value = mock_async_result

    repo_external_id = next(iter(autofix_full_finished_run.codebases.keys()))

    # Call the function
    response = run_autofix_push_changes(
        AutofixUpdateRequest(
            run_id=autofix_full_finished_run.run_id,
            payload=AutofixCreatePrUpdatePayload(repo_external_id=repo_external_id, make_pr=True),
        ),
    )

    # Assertions
    assert response.run_id == autofix_full_finished_run.run_id
    assert response.status == expected_status
    if expected_message:
        assert response.message == expected_message
    else:
        assert response.message is None

    # Check if get was called appropriately
    if scenario == "retry_success":
        assert mock_async_result.get.call_count == 2
    elif scenario == "success":
        assert mock_async_result.get.call_count == 1
    # In timeout scenarios, the number of calls depends on timing/backoff, so we just check the final result
