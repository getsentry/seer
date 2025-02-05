import os
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from seer.automation.autofix.steps.steps import AutofixPipelineStep
from seer.automation.state import DbStateRunTypes
from seer.automation.utils import make_kill_signal


class ConcreteAutofixPipelineStep(AutofixPipelineStep):
    @classmethod
    def get_task(cls):
        return MagicMock()

    @staticmethod
    def _instantiate_request(request):
        return MagicMock()

    @staticmethod
    def _instantiate_context(request, type: Optional["DbStateRunTypes"] = None):
        return MagicMock()

    def _invoke(self, **kwargs):
        return True

    def _handle_exception(self, exception):
        pass


class TestAutofixPipelineStep:
    @patch("seer.automation.autofix.steps.steps.make_done_signal", return_value="done:1")
    def test_pre_invoke(self, mock_make_done_signal):
        step = ConcreteAutofixPipelineStep({"run_id": 1, "step_id": 1})
        step.context.state.get = MagicMock(return_value=MagicMock(signals=[]))

        assert step._pre_invoke() is True

        step.context.state.get = MagicMock(return_value=MagicMock(signals=["done:1"]))
        assert step._pre_invoke() is False

        step._cleanup()  # _pre_invoke shouldn't be called standalone because it spawns a thread, so here we manually clean it up

    @patch("seer.automation.autofix.steps.steps.make_done_signal", return_value="done:1")
    def test_post_invoke(self, mock_make_done_signal):
        step = ConcreteAutofixPipelineStep({"run_id": 1, "step_id": 1})
        mock_state = MagicMock(signals=[])
        step.context.state.update = MagicMock()
        step.context.state.update.return_value.__enter__.return_value = mock_state

        step._post_invoke("result")

        step.context.state.update.assert_called_once()
        assert mock_state.signals == ["done:1"]

    @pytest.fixture
    def mock_step(self):
        class MockStep(AutofixPipelineStep):
            @classmethod
            def get_task(cls):
                return MagicMock()

            def _instantiate_request(self, request, type: Optional["DbStateRunTypes"] = None):
                return MagicMock(**request)

            def _instantiate_context(self, request, type: Optional["DbStateRunTypes"] = None):
                return MagicMock()

            def _invoke(self, **kwargs):
                return True

        return MockStep({"run_id": 1, "step_id": 1})

    def test_max_retries_default(self, mock_step):
        assert mock_step.max_retries == 0

    def test_get_retry_count_no_retries(self, mock_step):
        mock_step.context.signals = []
        assert mock_step.get_retry_count() == 0

    def test_get_retry_count_with_retries(self, mock_step):
        mock_step.context.signals = ["retry:1:1", "retry:1:2"]
        assert mock_step.get_retry_count() == 2

    def test_get_retry_count_with_mixed_signals(self, mock_step):
        mock_step.context.signals = ["retry:1:1", "other_signal", "retry:1:2"]
        assert mock_step.get_retry_count() == 2

    @patch("seer.automation.autofix.steps.steps.make_retry_prefix")
    def test_get_retry_count_with_custom_prefix(self, mock_make_retry_prefix, mock_step):
        mock_make_retry_prefix.return_value = "custom_retry:"
        mock_step.context.signals = ["custom_retry:1", "custom_retry:2", "other_signal"]
        assert mock_step.get_retry_count() == 2

    @patch("seer.automation.autofix.steps.steps.make_retry_prefix")
    @patch("seer.automation.autofix.steps.steps.make_retry_signal")
    def test_handle_exception_no_retry(
        self, mock_make_retry_signal, mock_make_retry_prefix, mock_step
    ):
        mock_make_retry_prefix.return_value = "retry:1:"
        mock_step.context.signals = []
        mock_step.context.event_manager.on_error = MagicMock()
        mock_step.next = MagicMock()

        exception = Exception("Test error")
        mock_step._handle_exception(exception)

        mock_step.context.event_manager.on_error.assert_called_once_with(
            "Oops, something went wrong inside. We use Sentry too, so we're already on it."
        )
        mock_step.next.assert_not_called()

    @patch("seer.automation.autofix.steps.steps.make_retry_prefix")
    @patch("seer.automation.autofix.steps.steps.make_retry_signal")
    def test_handle_exception_with_retry(
        self, mock_make_retry_signal, mock_make_retry_prefix, mock_step
    ):
        mock_make_retry_prefix.return_value = "retry:1:"
        mock_make_retry_signal.return_value = "retry:1:1"
        mock_step.max_retries = 1
        mock_step.context.signals = []
        mock_step.context.state.update = MagicMock()
        mock_step.context.state.update.return_value.__enter__.return_value = MagicMock(signals=[])
        mock_step.next = MagicMock()

        exception = Exception("Test error")
        mock_step._handle_exception(exception)

        mock_make_retry_signal.assert_called_once_with(1, 1)
        mock_step.next.assert_called_once()
        mock_step.context.event_manager.on_error.assert_called_once_with(
            str(exception), should_completely_error=False
        )

    @patch("seer.automation.autofix.steps.steps.make_retry_prefix")
    @patch("seer.automation.autofix.steps.steps.make_retry_signal")
    def test_handle_exception_max_retries_reached(
        self, mock_make_retry_signal, mock_make_retry_prefix, mock_step
    ):
        mock_make_retry_prefix.return_value = "retry:1:"
        mock_step.max_retries = 1
        mock_step.context.signals = ["retry:1:1"]
        mock_step.context.event_manager.on_error = MagicMock()
        mock_step.next = MagicMock()

        exception = Exception("Test error")
        mock_step._handle_exception(exception)

        mock_step.context.event_manager.on_error.assert_called_once_with(
            "Oops, something went wrong inside. We use Sentry too, so we're already on it."
        )
        mock_step.next.assert_not_called()

    @patch("os._exit")
    @patch("time.sleep")
    def test_check_for_kill_no_kill_signal(self, mock_sleep, mock_exit, mock_step):
        mock_step.thread_kill = False
        mock_step.context.state.get.return_value = MagicMock(signals=[])

        def set_thread_kill():
            time.sleep(0.3)
            mock_step.thread_kill = True

        threading.Thread(target=set_thread_kill).start()

        mock_step._check_for_kill()

        mock_exit.assert_not_called()
        assert mock_sleep.call_count >= 1

    @patch("os._exit")
    @patch("time.sleep")
    @patch.dict(os.environ, {})
    def test_check_for_kill_with_kill_signal(self, mock_sleep, mock_exit, mock_step):
        mock_step.thread_kill = False
        kill_signal = make_kill_signal()
        mock_step.context.state.get.return_value = MagicMock(signals=[kill_signal])
        mock_step.context.state.update.return_value.__enter__.return_value = MagicMock(
            signals=[kill_signal]
        )

        mock_step._check_for_kill()

        mock_exit.assert_called_once_with(1)
        assert mock_step.thread_kill is True

    @patch("os._exit")
    @patch("time.sleep")
    def test_check_for_kill_remove_kill_signal(self, mock_sleep, mock_exit, mock_step):
        mock_step.thread_kill = False
        kill_signal = make_kill_signal()
        mock_step.context.state.get.return_value = MagicMock(signals=[kill_signal])
        mock_state = MagicMock(signals=[kill_signal])
        mock_step.context.state.update.return_value.__enter__.return_value = mock_state

        mock_step._check_for_kill()

        assert kill_signal not in mock_state.signals
        mock_exit.assert_called_once_with(1)
        assert mock_step.thread_kill is True

    @patch("threading.Thread")
    def test_pre_invoke_starts_check_for_kill_thread(self, mock_thread, mock_step):
        mock_step._pre_invoke()

        mock_thread.assert_called_once_with(target=mock_step._check_for_kill)
        mock_thread.return_value.start.assert_called_once()

    def test_cleanup_sets_thread_kill(self, mock_step):
        mock_step.thread = MagicMock()
        mock_step.thread_kill = False
        mock_step._cleanup()

        assert mock_step.thread_kill is True
