import unittest
from unittest.mock import MagicMock, patch

from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.models import CodePrReviewRequest
from seer.automation.codegen.pr_review_step import PrReviewStep, PrReviewStepRequest
from seer.automation.models import RepoDefinition


class TestPrReview(unittest.TestCase):
    @patch("seer.automation.codegen.pr_review_coding_component.PrReviewCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_invoke_with_non_codecov_request(
        self, mock_instantiate_context, _, mock_invoke_pr_review_component
    ):
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
            "is_codecov_request": False,
        }
        request = PrReviewStepRequest(**request_data)
        step = PrReviewStep(request=request)
        step.context = mock_context

        step.invoke()

        mock_context.get_repo_client.assert_called_once_with(
            repo_name="owner1/repo1", type=RepoClientType.WRITE
        )
        mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
        mock_repo_client.get_pr_diff_content.assert_called_once_with(mock_pr.url)

        actual_request = mock_invoke_pr_review_component.call_args[0][0]
        assert isinstance(actual_request, CodePrReviewRequest)
        assert actual_request.diff == mock_diff_content

        # Verify is_codecov_request is passed correctly
        mock_invoke_pr_review_component.assert_called_once()
        mock_invoke_pr_review_component.assert_called_with(actual_request, is_codecov_request=False)

    @patch("seer.automation.codegen.pr_review_coding_component.PrReviewCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_invoke_with_codecov_request(
        self, mock_instantiate_context, _, mock_invoke_pr_review_component
    ):
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
            "is_codecov_request": True,
        }
        request = PrReviewStepRequest(**request_data)
        step = PrReviewStep(request=request)
        step.context = mock_context

        step.invoke()

        mock_context.get_repo_client.assert_called_once_with(
            repo_name="owner1/repo1", type=RepoClientType.CODECOV_PR_REVIEW
        )
        mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
        mock_repo_client.get_pr_diff_content.assert_called_once_with(mock_pr.url)

        actual_request = mock_invoke_pr_review_component.call_args[0][0]
        assert isinstance(actual_request, CodePrReviewRequest)
        assert actual_request.diff == mock_diff_content

        # Verify is_codecov_request is passed correctly
        mock_invoke_pr_review_component.assert_called_once()
        mock_invoke_pr_review_component.assert_called_with(actual_request, is_codecov_request=True)
