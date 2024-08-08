from unittest.mock import MagicMock, patch

import pytest
from johen import generate

from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixCreatePrUpdatePayload,
    AutofixRequest,
    AutofixRootCauseUpdatePayload,
    AutofixStatus,
    AutofixUpdateRequest,
)
from seer.automation.autofix.tasks import (
    get_autofix_state,
    get_autofix_state_from_pr_id,
    run_autofix_create_pr,
    run_autofix_execution,
    run_autofix_root_cause,
)
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
        mock_state.get.return_value = MagicMock(run_id=1, status=AutofixStatus.PENDING)
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
        mock_state.get.return_value = MagicMock(run_id=1, status=AutofixStatus.PENDING)
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
