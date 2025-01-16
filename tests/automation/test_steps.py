import unittest
from typing import Optional
from unittest.mock import MagicMock

from seer.automation.state import DbStateRunTypes
from src.seer.automation.steps import (
    ConditionalStep,
    ConditionalStepRequest,
    ParallelizedChainConditionalStep,
    ParallelizedChainStep,
    ParallelizedChainStepRequest,
)


class ConcreteConditionalStep(ConditionalStep):
    @classmethod
    def get_task(cls):
        return MagicMock()

    @staticmethod
    def _instantiate_request(request):
        return ConditionalStepRequest(**request)

    @staticmethod
    def _instantiate_context(request, type: Optional["DbStateRunTypes"] = None):
        return MagicMock()

    def condition(self):
        return True


class TestConditionalStep(unittest.TestCase):
    def test_condition_true(self):
        step = ConcreteConditionalStep(
            {
                "run_id": 1,
                "step_id": 1,
                "on_success": "success_signature",
                "on_failure": "failure_signature",
            }
        )

        step.next = MagicMock()
        step.condition = MagicMock(return_value=True)

        step.invoke()

        step.next.assert_called_once_with("success_signature")

    def test_condition_false(self):
        step = ConcreteConditionalStep(
            {
                "run_id": 1,
                "step_id": 1,
                "on_success": "success_signature",
                "on_failure": "failure_signature",
            }
        )

        step.next = MagicMock()
        step.condition = MagicMock(return_value=False)

        step.invoke()

        step.next.assert_called_once_with("failure_signature")


class ConcreteParallelizedChainStep(ParallelizedChainStep):
    @classmethod
    def get_task(cls):
        return MagicMock()

    @staticmethod
    def _instantiate_request(request):
        return ParallelizedChainStepRequest(**request)

    @staticmethod
    def _instantiate_context(request, type: Optional["DbStateRunTypes"] = None):
        return MagicMock()

    @staticmethod
    def _get_conditional_step_class() -> type[ParallelizedChainConditionalStep]:
        return ParallelizedChainConditionalStep

    def _handle_exception(self, exception: Exception):
        pass


class TestParallelizedChainStep(unittest.TestCase):
    def test_parallel_chain_scheduling(self):
        step = ConcreteParallelizedChainStep(
            {
                "run_id": 1,
                "steps": [
                    "signature_1",
                    "signature_2",
                ],
                "on_success": "final_success_signature",
            }
        )

        # Mock the _get_conditional_step_class to return a MagicMock with a specific get_signature method

        mock_generated_signature_1 = MagicMock()
        mock_generated_signature_1.kwargs = {"request": {"step_id": "1"}}
        mock_generated_signature_2 = MagicMock()
        mock_generated_signature_2.kwargs = {"request": {"step_id": "2"}}
        step.instantiate_signature = MagicMock(
            side_effect=[mock_generated_signature_1, mock_generated_signature_2]
        )

        mock_conditional_step_class = MagicMock()
        mock_conditional_step_class.get_signature = MagicMock(return_value="conditional_signature")
        step._get_conditional_step_class = MagicMock(return_value=mock_conditional_step_class)

        step.next = MagicMock()

        step.invoke()

        step.next.assert_any_call(mock_generated_signature_1, link="conditional_signature")
        step.next.assert_any_call(mock_generated_signature_2, link="conditional_signature")
