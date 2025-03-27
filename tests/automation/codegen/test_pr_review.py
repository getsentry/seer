import unittest
from unittest.mock import MagicMock, patch

from seer.automation.codegen.models import CodePrReviewOutput, CodePrReviewRequest
from seer.automation.codegen.pr_review_coding_component import PrReviewCodingComponent
from seer.automation.codegen.pr_review_step import PrReviewStep, PrReviewStepRequest
from seer.automation.models import RepoDefinition


class TestPrReview(unittest.TestCase):
    @patch("seer.automation.codegen.pr_review_coding_component.PrReviewCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_invoke_happy_path(self, mock_instantiate_context, _, mock_invoke_unit_test_component):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_diff_content = "diff content"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content

        request_data = {
            "run_id": 1,
            "pr_id": 123,
            "repo_definition": RepoDefinition(
                name="repo1", owner="owner1", provider="github", external_id="123123"
            ),
        }
        request = PrReviewStepRequest(**request_data)
        step = PrReviewStep(request=request)
        step.context = mock_context

        step.invoke()

        mock_context.get_repo_client.assert_called_once()
        mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
        mock_repo_client.get_pr_diff_content.assert_called_once_with(mock_pr.url)

        actual_request = mock_invoke_unit_test_component.call_args[0][0]

        assert isinstance(actual_request, CodePrReviewRequest)
        assert actual_request.diff == mock_diff_content
