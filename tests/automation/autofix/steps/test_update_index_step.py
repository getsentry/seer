import unittest
from unittest.mock import MagicMock

from seer.automation.autofix.steps.update_index_step import UpdateIndexStep


class TestUpdateIndexStep(unittest.TestCase):
    def test_update_index_step_happy_path(self):
        mock_context = MagicMock()
        mock_codebase = MagicMock()
        mock_codebase.repo_info.id = 1333333
        mock_context.codebases = {"1": mock_codebase}
        UpdateIndexStep._instantiate_context = MagicMock(return_value=mock_context)

        step = UpdateIndexStep({"run_id": 1, "step_id": 1, "repo_id": 1333333})

        step.invoke()

        mock_codebase.update.assert_called_once()
