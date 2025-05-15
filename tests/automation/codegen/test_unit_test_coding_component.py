import unittest
from unittest.mock import MagicMock, patch

from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodeUnitTestRequest
from seer.automation.codegen.unit_test_coding_component import UnitTestCodingComponent


class TestUnitTestCodingComponent(unittest.TestCase):
    def setUp(self):
        self.mock_context = MagicMock(spec=CodegenContext)
        self.component = UnitTestCodingComponent(self.mock_context)

    def test_get_client_type_with_codecov_request(self):
        client_type = self.component._get_client_type(is_codecov_request=True)
        self.assertEqual(client_type, RepoClientType.CODECOV_PR_REVIEW)

    def test_get_client_type_without_codecov_request(self):
        client_type = self.component._get_client_type(is_codecov_request=False)
        self.assertEqual(client_type, RepoClientType.CODECOV_UNIT_TEST)

    @patch("seer.automation.codegen.unit_test_coding_component.BaseTools")
    @patch("seer.automation.codegen.unit_test_coding_component.LlmAgent")
    @patch("seer.automation.codegen.unit_test_coding_component.CodecovClient")
    @patch("seer.automation.codegen.unit_test_coding_component.PlanStepsPromptXml")
    @patch("seer.automation.codegen.unit_test_coding_component.extract_text_inside_tags")
    def test_invoke_uses_correct_client_type_for_codecov_request(
        self,
        mock_extract_text,
        mock_plan_steps,
        mock_codecov_client,
        mock_agent,
        mock_base_tools,
    ):
        mock_request = MagicMock(spec=CodeUnitTestRequest)
        mock_request.diff = "test diff content"
        mock_request.codecov_client_params = {
            "repo_name": "repo",
            "pullid": 123,
            "owner_username": "owner",
            "head_sha": "sha",
        }
        mock_extract_text.return_value = "plan steps content"
        mock_plan_steps.from_xml().to_model().tasks = [MagicMock()]
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.run.return_value = "response"

        self.component.invoke(mock_request, is_codecov_request=True)

        mock_base_tools.assert_called_once()
        self.assertEqual(
            mock_base_tools.call_args[1]["repo_client_type"], RepoClientType.CODECOV_PR_REVIEW
        )

    @patch("seer.automation.codegen.unit_test_coding_component.BaseTools")
    @patch("seer.automation.codegen.unit_test_coding_component.LlmAgent")
    @patch("seer.automation.codegen.unit_test_coding_component.CodecovClient")
    @patch("seer.automation.codegen.unit_test_coding_component.PlanStepsPromptXml")
    @patch("seer.automation.codegen.unit_test_coding_component.extract_text_inside_tags")
    def test_invoke_uses_correct_client_type_for_non_codecov_request(
        self,
        mock_extract_text,
        mock_plan_steps,
        mock_codecov_client,
        mock_agent,
        mock_base_tools,
    ):
        """Test that invoke uses the correct client type when is_codecov_request is False"""
        mock_request = MagicMock(spec=CodeUnitTestRequest)
        mock_request.diff = "test diff content"
        mock_request.codecov_client_params = {
            "repo_name": "repo",
            "pullid": 123,
            "owner_username": "owner",
            "head_sha": "sha",
        }
        mock_extract_text.return_value = "plan steps content"
        mock_plan_steps.from_xml().to_model().tasks = [MagicMock()]
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.run.return_value = "response"

        self.component.invoke(mock_request, is_codecov_request=False)

        mock_base_tools.assert_called_once()
        self.assertEqual(
            mock_base_tools.call_args[1]["repo_client_type"], RepoClientType.CODECOV_UNIT_TEST
        )
