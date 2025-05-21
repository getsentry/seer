import unittest
from unittest.mock import MagicMock, patch

from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.models import CodePrReviewOutput, CodePrReviewRequest
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
        is_codecov_param = mock_invoke_pr_review_component.call_args[1].get("is_codecov_request")
        assert is_codecov_param is False

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
        is_codecov_param = mock_invoke_pr_review_component.call_args[1].get("is_codecov_request")
        assert is_codecov_param is True

    @patch("seer.automation.codegen.pr_review_coding_component.PrReviewCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_handle_no_changes_required_when_value_error(
        self, mock_instantiate_context, _, mock_invoke_pr_review_component
    ):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_diff_content = "diff content"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content

        # Have the component invoke raise a ValueError
        mock_invoke_pr_review_component.side_effect = ValueError("Test error")

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

        # Verify that publish_no_changes_required is called
        mock_repo_client.publish_no_changes_required.assert_called_once()

    @patch("seer.automation.codegen.pr_review_coding_component.PrReviewCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_handle_empty_generated_output(
        self, mock_instantiate_context, _, mock_invoke_pr_review_component
    ):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_diff_content = "diff content"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content

        # Component returns None (no changes required)
        mock_invoke_pr_review_component.return_value = None

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

        # Verify that publish_no_changes_required is called
        mock_repo_client.publish_no_changes_required.assert_called_once()

    @patch("seer.automation.codegen.pr_review_coding_component.PrReviewCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_successful_pr_review_publication(
        self, mock_instantiate_context, _, mock_invoke_pr_review_component
    ):
        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_diff_content = "diff content"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content

        # Component returns a valid PR review
        mock_generated_review = MagicMock(spec=CodePrReviewOutput)
        mock_invoke_pr_review_component.return_value = mock_generated_review

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

        # Verify that publish_generated_pr_review is called with the correct review
        mock_repo_client.publish_generated_pr_review.assert_called_once_with(
            pr_review=mock_generated_review
        )

        # Verify event manager methods were called
        mock_context.event_manager.mark_running.assert_called_once()
        mock_context.event_manager.mark_completed.assert_called_once()
