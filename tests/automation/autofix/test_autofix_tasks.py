import unittest
from unittest.mock import Mock, patch

from johen import generate

from seer.automation.autofix.models import AutofixContinuation, AutofixPrEventRequest
from seer.automation.autofix.tasks import get_autofix_state_from_pr_id, run_autofix_log_pr_event
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session


class TestGetStateFromPr(unittest.TestCase):
    def test_successful_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 1)
        self.assertIsNotNone(retrieved_state)
        if retrieved_state is not None:
            self.assertEqual(retrieved_state.get(), state)

    def test_no_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 2)
        self.assertIsNone(retrieved_state)


class TestRunAutofixLogPrEvent(unittest.TestCase):
    @patch("seer.automation.autofix.tasks.Langfuse")
    def test_valid_action_opened(self, mock_Langfuse):
        request = AutofixPrEventRequest(action="opened", run_id=123)
        mock_Langfuse.return_value.client.trace.list.return_value.data = [
            Mock(id="trace1", tags=["existing_tag"])
        ]

        run_autofix_log_pr_event(request)

        mock_Langfuse.return_value.client.trace.list.assert_called_once_with(tags=["run_id:123"])
        trace_call = mock_Langfuse.return_value.trace.call_args
        assert trace_call[1]["id"] == "trace1"
        assert set(trace_call[1]["tags"]) == set(["existing_tag", "pr:opened"])

    @patch("seer.automation.autofix.tasks.Langfuse")
    def test_valid_action_closed(self, mock_Langfuse):
        request = AutofixPrEventRequest(action="closed", run_id=456)
        mock_Langfuse.return_value.client.trace.list.return_value.data = [
            Mock(id="trace2", tags=None)
        ]

        run_autofix_log_pr_event(request)

        mock_Langfuse.return_value.client.trace.list.assert_called_once_with(tags=["run_id:456"])
        trace_call = mock_Langfuse.return_value.trace.call_args
        assert trace_call[1]["id"] == "trace2"
        assert set(trace_call[1]["tags"]) == set(["pr:closed"])

    @patch("seer.automation.autofix.tasks.Langfuse")
    def test_valid_action_merged(self, mock_Langfuse):
        request = AutofixPrEventRequest(action="merged", run_id=789)
        mock_Langfuse.return_value.client.trace.list.return_value.data = [
            Mock(id="trace3", tags=["tag1", "tag2"])
        ]

        run_autofix_log_pr_event(request)

        mock_Langfuse.return_value.client.trace.list.assert_called_once_with(tags=["run_id:789"])
        trace_call = mock_Langfuse.return_value.trace.call_args
        assert trace_call[1]["id"] == "trace3"
        assert set(trace_call[1]["tags"]) == set(["tag1", "tag2", "pr:merged"])

    @patch("seer.automation.autofix.tasks.Langfuse")
    def test_multiple_traces(self, mock_Langfuse):
        request = AutofixPrEventRequest(action="opened", run_id=1000)
        mock_Langfuse.return_value.client.trace.list.return_value.data = [
            Mock(id="trace4", tags=["tag1"]),
            Mock(id="trace5", tags=["tag2"]),
        ]

        run_autofix_log_pr_event(request)

        mock_Langfuse.return_value.client.trace.list.assert_called_once_with(tags=["run_id:1000"])
        trace_calls = mock_Langfuse.return_value.trace.call_args_list
        assert len(trace_calls) == 2
        assert trace_calls[0][1]["id"] == "trace4"
        assert set(trace_calls[0][1]["tags"]) == set(["tag1", "pr:opened"])
        assert trace_calls[1][1]["id"] == "trace5"
        assert set(trace_calls[1][1]["tags"]) == set(["tag2", "pr:opened"])
