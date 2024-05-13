from unittest.mock import MagicMock, patch

from seer.automation.autofix.steps.steps import AutofixPipelineStep


class ConcreteAutofixPipelineStep(AutofixPipelineStep):
    @classmethod
    def get_task(cls):
        return MagicMock()

    @staticmethod
    def _instantiate_request(request):
        return MagicMock()

    @staticmethod
    def _instantiate_context(request):
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

    @patch("seer.automation.autofix.steps.steps.make_done_signal", return_value="done:1")
    def test_post_invoke(self, mock_make_done_signal):
        step = ConcreteAutofixPipelineStep({"run_id": 1, "step_id": 1})
        mock_state = MagicMock(signals=[])
        step.context.state.update = MagicMock()
        step.context.state.update.return_value.__enter__.return_value = mock_state

        step._post_invoke("result")

        step.context.state.update.assert_called_once()
        assert mock_state.signals == ["done:1"]
