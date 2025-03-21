import unittest
from unittest.mock import MagicMock, patch

from langfuse.decorators import observe

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient
from seer.automation.autofix.components.coding.models import PlanStepsPromptXml
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodeUnitTestOutput, CodeUnitTestRequest
from seer.automation.codegen.unit_test_coding_component import UnitTestCodingComponent
from seer.automation.models import FileChange


class TestUnitTestCodingComponent(unittest.TestCase):
    @patch("seer.automation.codegen.unit_test_coding_component.extract_text_inside_tags")
    @patch("seer.automation.codegen.unit_test_coding_component.BaseTools")
    @patch("seer.automation.codegen.unit_test_coding_component.LlmAgent")
    @patch("seer.automation.codegen.unit_test_coding_component.CodecovClient")
    def test_invoke_with_default_client_type(
        self, mock_codecov_client, mock_llm_agent, mock_base_tools, mock_extract
    ):
        # Setup
        mock_context = MagicMock(spec=CodegenContext)
        mock_repo_client = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client

        mock_tools = MagicMock()
        mock_tools.get_tools.return_value = ["tool1", "tool2"]
        mock_base_tools.return_value.__enter__.return_value = mock_tools

        mock_agent = MagicMock(spec=LlmAgent)
        mock_agent.run.return_value = "final_response"
        mock_agent.memory = {"memory": "data"}
        mock_llm_agent.return_value = mock_agent

        mock_extract.return_value = "plan_steps_content"

        mock_plan_steps = MagicMock()
        mock_task = MagicMock()
        mock_task.type = "file_change"
        mock_task.file_path = "test/path.py"
        mock_task.repo_name = "repo1"
        mock_plan_steps.tasks = [mock_task]

        mock_codecov_client.fetch_coverage.return_value = {"coverage": "data"}
        mock_codecov_client.fetch_test_results_for_commit.return_value = {"test_results": "data"}

        mock_repo_client.get_file_content.return_value = ("file content", "utf-8")

        with patch(
            "seer.automation.codegen.unit_test_coding_component.PlanStepsPromptXml.from_xml"
        ) as mock_from_xml:
            mock_from_xml.return_value.to_model.return_value = mock_plan_steps

            # Create component and request
            component = UnitTestCodingComponent(mock_context)
            request = CodeUnitTestRequest(
                diff="test diff",
                codecov_client_params={
                    "repo_name": "repo1",
                    "pullid": 123,
                    "owner_username": "owner1",
                    "head_sha": "sha123",
                },
            )
            mock_llm_client = MagicMock(spec=LlmClient)

            # Execute
            result = component.invoke(request, is_codecov_request=False, llm_client=mock_llm_client)

            # Verify
            self.assertIsNotNone(result)
            self.assertIsInstance(result, CodeUnitTestOutput)
            mock_base_tools.assert_called_once_with(
                mock_context, repo_client_type=RepoClientType.CODECOV_UNIT_TEST
            )
            mock_context.get_repo_client.assert_called_once_with(
                mock_task.repo_name, type=RepoClientType.CODECOV_UNIT_TEST
            )
            mock_context.store_memory.assert_called_once_with("unit_test_memory", mock_agent.memory)

    @patch("seer.automation.codegen.unit_test_coding_component.extract_text_inside_tags")
    @patch("seer.automation.codegen.unit_test_coding_component.BaseTools")
    @patch("seer.automation.codegen.unit_test_coding_component.LlmAgent")
    @patch("seer.automation.codegen.unit_test_coding_component.CodecovClient")
    def test_invoke_with_codecov_request_true(
        self, mock_codecov_client, mock_llm_agent, mock_base_tools, mock_extract
    ):
        # Setup
        mock_context = MagicMock(spec=CodegenContext)
        mock_repo_client = MagicMock()
        mock_context.get_repo_client.return_value = mock_repo_client

        mock_tools = MagicMock()
        mock_tools.get_tools.return_value = ["tool1", "tool2"]
        mock_base_tools.return_value.__enter__.return_value = mock_tools

        mock_agent = MagicMock(spec=LlmAgent)
        mock_agent.run.return_value = "final_response"
        mock_agent.memory = {"memory": "data"}
        mock_llm_agent.return_value = mock_agent

        mock_extract.return_value = "plan_steps_content"

        mock_plan_steps = MagicMock()
        mock_task = MagicMock()
        mock_task.type = "file_change"
        mock_task.file_path = "test/path.py"
        mock_task.repo_name = "repo1"
        mock_plan_steps.tasks = [mock_task]

        mock_codecov_client.fetch_coverage.return_value = {"coverage": "data"}
        mock_codecov_client.fetch_test_results_for_commit.return_value = {"test_results": "data"}

        mock_repo_client.get_file_content.return_value = ("file content", "utf-8")

        with patch(
            "seer.automation.codegen.unit_test_coding_component.PlanStepsPromptXml.from_xml"
        ) as mock_from_xml:
            mock_from_xml.return_value.to_model.return_value = mock_plan_steps

            # Create component and request
            component = UnitTestCodingComponent(mock_context)
            request = CodeUnitTestRequest(
                diff="test diff",
                codecov_client_params={
                    "repo_name": "repo1",
                    "pullid": 123,
                    "owner_username": "owner1",
                    "head_sha": "sha123",
                },
            )
            mock_llm_client = MagicMock(spec=LlmClient)

            # Execute
            result = component.invoke(request, is_codecov_request=True, llm_client=mock_llm_client)

            # Verify
            self.assertIsNotNone(result)
            mock_context.get_repo_client.assert_called_once_with(
                mock_task.repo_name, type=RepoClientType.CODECOV_PR_REVIEW
            )
