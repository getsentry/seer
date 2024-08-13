from unittest.mock import MagicMock, patch

import pytest

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.autofix.state import ContinuationState
from seer.db import DbRunState


class TestContinuationState:
    @pytest.fixture
    def mock_session(self):
        with patch("seer.automation.autofix.state.Session") as mock:
            yield mock

    def test_from_id(self):
        with patch("seer.automation.state.DbState.from_id") as mock_from_id:
            mock_from_id.return_value = MagicMock(spec=ContinuationState)
            result = ContinuationState.from_id(1, AutofixContinuation)

            mock_from_id.assert_called_once_with(1, AutofixContinuation)
            assert isinstance(result, ContinuationState)

    def test_set(self, mock_session):
        mock_session_instance = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_session_instance

        state = ContinuationState(id=1, model=AutofixContinuation)
        mock_autofix_continuation = MagicMock(spec=AutofixContinuation)
        mock_autofix_continuation.model_dump.return_value = {"key": "value"}
        mock_autofix_continuation.updated_at = "2023-01-01"
        mock_autofix_continuation.last_triggered_at = "2023-01-02"

        state.set(mock_autofix_continuation)

        mock_autofix_continuation.mark_updated.assert_called_once()
        mock_session_instance.merge.assert_called_once()
        mock_session_instance.commit.assert_called_once()

        args, _ = mock_session_instance.merge.call_args
        db_state = args[0]
        assert isinstance(db_state, DbRunState)
        assert db_state.id == 1
        assert db_state.value == {"key": "value"}
        assert db_state.updated_at == "2023-01-01"
        assert db_state.last_triggered_at == "2023-01-02"

    def test_get(self):
        with patch("seer.automation.state.DbState.get") as mock_get:
            mock_get.return_value = MagicMock(spec=AutofixContinuation)
            state = ContinuationState(id=1, model=AutofixContinuation)
            result = state.get()

            mock_get.assert_called_once()
            assert isinstance(result, AutofixContinuation)

    def test_update(self):
        with patch("seer.automation.state.DbState.update") as mock_update:
            mock_context = MagicMock()
            mock_update.return_value.__enter__.return_value = mock_context

            state = ContinuationState(id=1, model=AutofixContinuation)
            with state.update() as context:
                assert context == mock_context

            mock_update.assert_called_once()
