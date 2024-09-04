import datetime
from unittest.mock import MagicMock, call, patch

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
    check_and_mark_recent_autofix_runs,
    delete_all_runs_before,
    delete_all_summaries_before,
    delete_data_for_ttl,
    get_autofix_state,
    get_autofix_state_from_pr_id,
    run_autofix_create_pr,
    run_autofix_execution,
    run_autofix_root_cause,
)
from seer.db import DbIssueSummary, DbPrIdToAutofixRunIdMapping, DbRunState, Session


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


class TestDeleteOldAutofixRuns:
    def test_old_data_is_deleted(self):
        old_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
            value="test_value",
        )
        old_summary = DbIssueSummary(
            group_id=1,
            summary="summary",
            created_at=datetime.datetime.now() - datetime.timedelta(days=100),
        )
        with Session() as session:
            session.add(old_run)
            session.add(old_summary)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_run_count = delete_all_runs_before(before_date)
        deleted_summary_count = delete_all_summaries_before(before_date)

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == old_run.id).first()
            remaining_summary = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.group_id == old_summary.group_id)
                .first()
            )
        assert remaining_run is None
        assert remaining_summary is None
        assert deleted_run_count == 1
        assert deleted_summary_count == 1

    def test_data_not_deleted_within_ttl(self):
        recent_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=80),
            value="test_value",
        )
        recent_summary = DbIssueSummary(
            group_id=1,
            summary="summary",
            created_at=datetime.datetime.now() - datetime.timedelta(days=80),
        )
        with Session() as session:
            session.add(recent_run)
            session.add(recent_summary)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_run_count = delete_all_runs_before(before_date)
        deleted_summary_count = delete_all_summaries_before(before_date)

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == recent_run.id).first()
            remaining_summary = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.group_id == recent_summary.group_id)
                .first()
            )
        assert remaining_run is not None
        assert remaining_summary is not None
        assert remaining_run.last_triggered_at > before_date
        assert remaining_summary.created_at > before_date
        assert deleted_run_count == 0
        assert deleted_summary_count == 0

    def test_batch_delete(self):
        old_runs = [
            DbRunState(
                last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
                value="test_value",
            )
            for _ in range(25)
        ]
        old_summaries = [
            DbIssueSummary(
                group_id=i,
                summary="summary",
                created_at=datetime.datetime.now() - datetime.timedelta(days=100),
            )
            for i in range(35)
        ]
        with Session() as session:
            session.bulk_save_objects(old_runs)
            session.bulk_save_objects(old_summaries)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_run_count = delete_all_runs_before(before_date, batch_size=10)
        deleted_summary_count = delete_all_summaries_before(before_date, batch_size=10)

        with Session() as session:
            remaining_runs = (
                session.query(DbRunState).filter(DbRunState.last_triggered_at < before_date).count()
            )
            remaining_summaries = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.created_at < before_date)
                .count()
            )
        assert remaining_runs == 0
        assert remaining_summaries == 0
        assert deleted_run_count == 25
        assert deleted_summary_count == 35

    def test_delete_old_autofix_runs_task(self):
        old_run = DbRunState(
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
            value="test_value",
        )
        old_summary = DbIssueSummary(
            group_id=1,
            summary="summary",
            created_at=datetime.datetime.now() - datetime.timedelta(days=100),
        )
        with Session() as session:
            session.add(old_run)
            session.add(old_summary)
            session.commit()

        # Celery task
        delete_data_for_ttl()

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == old_run.id).first()
            remaining_summary = (
                session.query(DbIssueSummary)
                .filter(DbIssueSummary.group_id == old_summary.group_id)
                .first()
            )
        assert remaining_run is None
        assert remaining_summary is None

    def test_delete_with_mapped_pr(self):
        old_run = DbRunState(
            id=789,
            last_triggered_at=datetime.datetime.now() - datetime.timedelta(days=100),
            value="test_value",
        )
        mapped_pr = DbPrIdToAutofixRunIdMapping(
            provider="github",
            pr_id=123,
            run_id=789,
        )
        with Session() as session:
            session.add(old_run)
            session.add(mapped_pr)
            session.commit()

        before_date = datetime.datetime.now() - datetime.timedelta(days=90)
        deleted_count = delete_all_runs_before(before_date)

        with Session() as session:
            remaining_run = session.query(DbRunState).filter(DbRunState.id == old_run.id).first()
            remaining_pr = (
                session.query(DbPrIdToAutofixRunIdMapping)
                .filter(DbPrIdToAutofixRunIdMapping.run_id == old_run.id)
                .first()
            )
        assert remaining_run is None
        assert remaining_pr is None
        assert deleted_count == 1
