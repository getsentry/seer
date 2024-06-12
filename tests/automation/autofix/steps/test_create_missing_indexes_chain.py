import unittest
from unittest.mock import MagicMock, patch

from johen import generate

from seer.automation.autofix.models import AutofixRequest
from seer.automation.autofix.steps.create_missing_indexes_chain import CreateMissingIndexesStep
from seer.automation.models import RepoDefinition
from seer.automation.steps import ParallelizedChainStepRequest


class TestCreateMissingIndicesChain(unittest.TestCase):
    @patch("seer.automation.autofix.steps.create_missing_indexes_chain.CreateIndexStep")
    @patch(
        "seer.automation.autofix.steps.create_missing_indexes_chain.AutofixParallelizedChainStep"
    )
    def test_recreate_all_codebases(self, mock_AutofixParallelizedChainStep, mock_CreateIndexStep):
        mock_context = MagicMock()
        mock_request = next(generate(AutofixRequest))
        mock_context.state.get.return_value.request = mock_request
        mock_context.repos = [next(generate(RepoDefinition)) for _ in range(5)]

        mock_codebase = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.is_ready = MagicMock(return_value=False)
        mock_codebase.workspace = mock_workspace
        mock_context.get_codebase_from_external_id = MagicMock(return_value=mock_codebase)

        mock_CreateIndexStep.get_signature.return_value = "create_index"
        mock_AutofixParallelizedChainStep.get_signature.return_value = "autofix_parallelized_chain"

        CreateMissingIndexesStep._instantiate_context = MagicMock(return_value=mock_context)
        CreateMissingIndexesStep.next = MagicMock()

        CreateMissingIndexesStep({"run_id": 1, "step_id": 1, "next": "signature_1"}).invoke()

        mock_AutofixParallelizedChainStep.get_signature.called_once_with(
            ParallelizedChainStepRequest(
                run_id=1,
                steps=["create_index" for _ in range(5)],
                on_success="signature_1",
            )
        )
        CreateMissingIndexesStep.next.assert_called_once_with("autofix_parallelized_chain")

    @patch("seer.automation.autofix.steps.create_missing_indexes_chain.CreateIndexStep")
    @patch(
        "seer.automation.autofix.steps.create_missing_indexes_chain.AutofixParallelizedChainStep"
    )
    def test_recreate_one_codebase(self, mock_AutofixParallelizedChainStep, mock_CreateIndexStep):
        mock_context = MagicMock()
        mock_request = next(generate(AutofixRequest))
        mock_context.state.get.return_value.request = mock_request
        mock_context.repos = [next(generate(RepoDefinition)) for _ in range(5)]

        mock_codebase = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.is_ready = MagicMock(side_effect=[True, True, True, False, True])
        mock_codebase.workspace = mock_workspace
        mock_codebase.is_behind.return_value = False
        mock_context.get_codebase_from_external_id = MagicMock(return_value=mock_codebase)

        mock_CreateIndexStep.get_signature.return_value = "create_index"
        mock_AutofixParallelizedChainStep.get_signature.return_value = "autofix_parallelized_chain"

        CreateMissingIndexesStep._instantiate_context = MagicMock(return_value=mock_context)
        CreateMissingIndexesStep.next = MagicMock()

        CreateMissingIndexesStep({"run_id": 1, "step_id": 1, "next": "signature_1"}).invoke()

        mock_AutofixParallelizedChainStep.get_signature.called_once_with(
            ParallelizedChainStepRequest(
                run_id=1,
                steps=["create_index"],
                on_success="signature_1",
            )
        )
        CreateMissingIndexesStep.next.assert_called_once_with("autofix_parallelized_chain")
