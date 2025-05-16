import unittest
from unittest.mock import MagicMock, patch

from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodeUnitTestOutput, CodeUnitTestRequest
from seer.automation.codegen.unit_test_coding_component import UnitTestCodingComponent
from seer.automation.models import FileChange


class TestUnitTestCodingComponent(unittest.TestCase):
    def setUp(self):
        # Create mocks
        self.mock_context = MagicMock(spec=CodegenContext)
        self.component = UnitTestCodingComponent(self.mock_context)

        # Setup test data
        self.test_request = MagicMock(spec=CodeUnitTestRequest)
        self.test_request.diff = "test diff content"
        self.test_request.codecov_client_params = {
            "repo_name": "repo",
            "pullid": 123,
            "owner_username": "owner",
            "head_sha": "sha",
        }

    def test_get_client_type_with_codecov_request(self):
        client_type = self.component._get_client_type(is_codecov_request=True)
        self.assertEqual(client_type, RepoClientType.CODECOV_PR_REVIEW)

    def test_get_client_type_without_codecov_request(self):
        client_type = self.component._get_client_type(is_codecov_request=False)
        self.assertEqual(client_type, RepoClientType.CODECOV_UNIT_TEST)

    @patch("seer.automation.codegen.unit_test_coding_component.task_to_file_change")
    @patch("seer.automation.codegen.unit_test_coding_component.extract_text_inside_tags")
    @patch("seer.automation.codegen.unit_test_coding_component.PlanStepsPromptXml")
    @patch("seer.automation.codegen.unit_test_coding_component.CodecovClient")
    @patch("seer.automation.codegen.unit_test_coding_component.LlmAgent")
    @patch("seer.automation.codegen.unit_test_coding_component.BaseTools")
    def test_invoke_uses_correct_client_type_for_codecov_request(
        self,
        mock_base_tools,
        mock_agent,
        mock_codecov_client,
        mock_plan_steps,
        mock_extract_text,
        mock_task_to_file_change,
    ):
        # Configure mocks
        self._setup_mocks(
            mock_base_tools,
            mock_agent,
            mock_codecov_client,
            mock_plan_steps,
            mock_extract_text,
            mock_task_to_file_change,
        )

        mock_llm_client = MagicMock()
        mock_llm_client.generate_text.return_value = "LLM response"

        result = self.component.invoke(
            self.test_request, is_codecov_request=True, llm_client=mock_llm_client
        )

        mock_base_tools.assert_called_once()
        self.assertEqual(
            mock_base_tools.call_args[1]["repo_client_type"], RepoClientType.CODECOV_PR_REVIEW
        )

        self.mock_context.get_repo_client.assert_called_with(
            repo_name="repo", type=RepoClientType.CODECOV_PR_REVIEW
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, CodeUnitTestOutput)

    @patch("seer.automation.codegen.unit_test_coding_component.task_to_file_change")
    @patch("seer.automation.codegen.unit_test_coding_component.extract_text_inside_tags")
    @patch("seer.automation.codegen.unit_test_coding_component.PlanStepsPromptXml")
    @patch("seer.automation.codegen.unit_test_coding_component.CodecovClient")
    @patch("seer.automation.codegen.unit_test_coding_component.LlmAgent")
    @patch("seer.automation.codegen.unit_test_coding_component.BaseTools")
    def test_invoke_uses_correct_client_type_for_non_codecov_request(
        self,
        mock_base_tools,
        mock_agent,
        mock_codecov_client,
        mock_plan_steps,
        mock_extract_text,
        mock_task_to_file_change,
    ):
        # Configure mocks
        self._setup_mocks(
            mock_base_tools,
            mock_agent,
            mock_codecov_client,
            mock_plan_steps,
            mock_extract_text,
            mock_task_to_file_change,
        )

        mock_llm_client = MagicMock()
        mock_llm_client.generate_text.return_value = "LLM response"

        result = self.component.invoke(
            self.test_request, is_codecov_request=False, llm_client=mock_llm_client
        )

        mock_base_tools.assert_called_once()
        self.assertEqual(
            mock_base_tools.call_args[1]["repo_client_type"], RepoClientType.CODECOV_UNIT_TEST
        )

        self.mock_context.get_repo_client.assert_called_with(
            repo_name="repo", type=RepoClientType.CODECOV_UNIT_TEST
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, CodeUnitTestOutput)

    def _setup_mocks(
        self,
        mock_base_tools,
        mock_agent,
        mock_codecov_client,
        mock_plan_steps,
        mock_extract_text,
        mock_task_to_file_change,
    ):
        # Configure mock return values
        mock_tools_instance = MagicMock()
        mock_base_tools.return_value.__enter__.return_value = mock_tools_instance
        mock_tools_instance.get_tools.return_value = ["mocked_tool1", "mocked_tool2"]

        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.run.return_value = "Agent response"
        mock_agent_instance.memory = ["memory_item1", "memory_item2"]

        # Mock codecov client methods
        mock_codecov_client.fetch_coverage.return_value = {"mock": "coverage_data"}
        mock_codecov_client.fetch_test_results_for_commit.return_value = {"mock": "test_results"}

        # Mock extract_text_inside_tags and PlanStepsPromptXml
        mock_extract_text.return_value = "plan steps content"

        mock_model = MagicMock()
        mock_task = MagicMock()
        mock_task.repo_name = "repo"
        mock_task.type = "file_change"
        mock_task.file_path = "path/to/file.py"
        mock_model.tasks = [mock_task]
        mock_plan_steps.from_xml.return_value.to_model.return_value = mock_model

        # Mock task_to_file_change
        mock_task_to_file_change.return_value = ([MagicMock(spec=FileChange)], None)

        # Mock repo client and file content
        mock_repo_client = MagicMock()
        self.mock_context.get_repo_client.return_value = mock_repo_client
        mock_repo_client.get_file_content.return_value = ("file content", "path")
