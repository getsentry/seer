import unittest
from unittest.mock import MagicMock

from seer.automation.autofix.steps.create_index_step import CreateIndexStep
from seer.automation.models import RepoDefinition


class TestCreateIndexStep(unittest.TestCase):
    def test_create_index_step_happy_path(self):
        mock_context = MagicMock()
        CreateIndexStep._instantiate_context = MagicMock(return_value=mock_context)

        repo = RepoDefinition(
            provider="github",
            owner="test_owner",
            name="test_repo",
            external_id="2",
        )
        step = CreateIndexStep({"run_id": 1, "step_id": 1, "repo": repo.model_dump()})

        step.invoke()

        mock_context.create_codebase_index.assert_called_once_with(repo)
