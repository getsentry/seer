import unittest
from unittest.mock import patch, MagicMock

from seer.automation.codegen.models import CodeUnitTestRequest
from seer.automation.codegen.unittest_step import UnittestStep, UnittestStepRequest
from seer.automation.models import RepoDefinition

import unittest
from unittest.mock import MagicMock

from seer.automation.models import RepoDefinition

class TestUnittestStep(unittest.TestCase):
    @patch("seer.automation.codegen.unit_test_coding_component.UnitTestCodingComponent.invoke")
    @patch("seer.automation.pipeline.PipelineStep", new_callable=MagicMock)
    @patch("seer.automation.codegen.step.CodegenStep._instantiate_context", new_callable=MagicMock)
    def test_invoke_happy_path(self, 
                    mock_instantiate_context,
                    _, 
                    mock_invoke_unit_test_component):

        mock_repo_client = MagicMock()
        mock_pr = MagicMock()
        mock_diff_content = "diff content"
        mock_latest_commit_sha = "latest_commit_sha"
        mock_context = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.repo.get_pull.return_value = mock_pr
        mock_repo_client.get_pr_diff_content.return_value = mock_diff_content
        mock_repo_client.get_pr_head_sha.return_value = mock_latest_commit_sha

        request_data = {
            "run_id": 1,
            "pr_id": 123,
            "repo_definition": RepoDefinition(name="repo1", owner="owner1", provider="github", external_id="123123")
        }
        request = UnittestStepRequest(**request_data)
        step = UnittestStep(request=request)
        step.context = mock_context

        step.invoke()

        mock_context.get_repo_client.assert_called_once()
        mock_repo_client.repo.get_pull.assert_called_once_with(request.pr_id)
        mock_repo_client.get_pr_diff_content.assert_called_once_with(mock_pr.url)
        mock_repo_client.get_pr_head_sha.assert_called_once_with(mock_pr.url)

        actual_request = mock_invoke_unit_test_component.call_args[0][0]

        assert isinstance(actual_request, CodeUnitTestRequest)
        assert actual_request.diff == mock_diff_content
        assert actual_request.codecov_client_params == {
            "repo_name": request.repo_definition.name,
            "pullid": request.pr_id,
            "owner_username": request.repo_definition.owner,
            "head_sha": mock_latest_commit_sha,
        }
