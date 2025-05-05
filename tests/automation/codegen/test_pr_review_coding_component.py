import unittest
from unittest.mock import MagicMock, patch

from seer.automation.agent.agent import AgentConfig
from seer.automation.codegen.models import CodePrReviewOutput, CodePrReviewRequest
from seer.automation.codegen.pr_review_coding_component import PrReviewCodingComponent


class TestPrReviewCodingComponent(unittest.TestCase):
    def setUp(self):
        self.mock_context = MagicMock()
        self.mock_tools = MagicMock()
        self.mock_agent = MagicMock()
        self.mock_llm_client = MagicMock()

        # Create component with mocked context
        self.component = PrReviewCodingComponent(self.mock_context)

        # Mock request
        self.request = CodePrReviewRequest(diff="mock diff content")

        # Set up patches
        self.tools_patcher = patch("seer.automation.codegen.pr_review_coding_component.BaseTools")
        self.agent_patcher = patch("seer.automation.codegen.pr_review_coding_component.LlmAgent")

        # Start patches
        self.mock_tools_class = self.tools_patcher.start()
        self.mock_agent_class = self.agent_patcher.start()

        # Configure mock return values
        self.mock_tools_instance = self.mock_tools_class.return_value.__enter__.return_value
        self.mock_tools_instance.get_tools.return_value = ["mocked_tool1", "mocked_tool2"]

        self.mock_agent_instance = self.mock_agent_class.return_value
        self.mock_agent_instance.run.return_value = "Agent response"
        self.mock_agent_instance.memory = ["memory_item1", "memory_item2"]

    def tearDown(self):
        self.tools_patcher.stop()
        self.agent_patcher.stop()

    @patch("seer.automation.codegen.pr_review_coding_component.LlmClient")
    def test_invoke_returns_none_when_agent_run_returns_none(self, mock_llm_client_class):
        # Set up agent to return None
        self.mock_agent_instance.run.return_value = None

        # Call invoke
        result = self.component.invoke(self.request, mock_llm_client_class)

        # Assert result is None
        self.assertIsNone(result)

        # Verify agent was correctly initialized and run
        self.mock_agent_class.assert_called_once_with(
            tools=self.mock_tools_instance.get_tools.return_value,
            config=AgentConfig(interactive=False),
        )
        self.mock_agent_instance.run.assert_called_once()

    @patch("seer.automation.codegen.pr_review_coding_component.LlmClient")
    def test_invoke_returns_none_when_formatted_response_is_none(self, mock_llm_client_class):
        # Set up formatted_response to return None
        mock_llm_client_class.generate_structured.return_value = None

        # Call invoke
        result = self.component.invoke(self.request, mock_llm_client_class)

        # Assert result is None
        self.assertIsNone(result)

    @patch("seer.automation.codegen.pr_review_coding_component.LlmClient")
    def test_invoke_returns_none_when_formatted_response_parsed_is_none(
        self, mock_llm_client_class
    ):
        # Set up formatted_response with parsed=None
        mock_formatted_response = MagicMock()
        mock_formatted_response.parsed = None

        # First call returns pr_description, second call returns formatted_response
        mock_llm_client_class.generate_structured.side_effect = [
            MagicMock(parsed=MagicMock(spec=CodePrReviewOutput.PrDescription)),
            mock_formatted_response,
        ]

        # Call invoke
        result = self.component.invoke(self.request, mock_llm_client_class)

        # Assert result is None
        self.assertIsNone(result)

    @patch("seer.automation.codegen.pr_review_coding_component.LlmClient")
    def test_invoke_success_with_structured_pr_description(self, mock_llm_client_class):
        # Mock PR description
        mock_pr_description = MagicMock()
        mock_pr_description.parsed = MagicMock(spec=CodePrReviewOutput.PrDescription)
        mock_pr_description.parsed.purpose = "Test purpose"
        mock_pr_description.parsed.key_technical_changes = "Test technical changes"
        mock_pr_description.parsed.architecture_decisions = "Test architecture decisions"
        mock_pr_description.parsed.dependencies_and_interactions = "Test dependencies"
        mock_pr_description.parsed.risk_considerations = "Test risks"
        mock_pr_description.parsed.notable_implementation_details = "Test implementation details"

        # Mock formatted response with comments
        mock_comment = MagicMock(spec=CodePrReviewOutput.Comment)
        mock_comment.path = "test_file.py"
        mock_comment.line = 10
        mock_comment.body = "Test comment"
        mock_comment.start_line = 5
        mock_comment.suggestion = "Test suggestion"

        mock_formatted_response = MagicMock()
        mock_formatted_response.parsed = [mock_comment]

        # Configure LlmClient.generate_structured to return different responses on each call
        mock_llm_client_class.generate_structured.side_effect = [
            mock_pr_description,  # First call returns PR description
            mock_formatted_response,  # Second call returns formatted comments
        ]

        # Call invoke
        result = self.component.invoke(self.request, mock_llm_client_class)

        # Verify LlmClient.generate_structured was called correctly for PR description
        pr_description_call = mock_llm_client_class.generate_structured.call_args_list[0]
        self.assertEqual(pr_description_call[1]["messages"], self.mock_agent_instance.memory)
        self.assertEqual(
            pr_description_call[1]["response_format"], CodePrReviewOutput.PrDescription
        )
        self.assertEqual(pr_description_call[1]["run_name"], "Generate PR description")
        self.assertEqual(pr_description_call[1]["max_tokens"], 4096)

        # Verify LlmClient.generate_structured was called correctly for formatted comments
        formatted_response_call = mock_llm_client_class.generate_structured.call_args_list[1]
        self.assertEqual(formatted_response_call[1]["messages"], self.mock_agent_instance.memory)
        self.assertEqual(
            formatted_response_call[1]["response_format"], list[CodePrReviewOutput.Comment]
        )
        self.assertEqual(formatted_response_call[1]["run_name"], "Generate PR review structured")
        self.assertEqual(formatted_response_call[1]["max_tokens"], 8192)

        # Verify suggestion was added to comment body
        self.assertIn(f"\n```suggestion\n{mock_comment.suggestion}\n```", mock_comment.body)

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.comments, mock_formatted_response.parsed)
        self.assertEqual(result.description, mock_pr_description.parsed)

    @patch("seer.automation.codegen.pr_review_coding_component.LlmClient")
    def test_invoke_success_with_comments_without_suggestions(self, mock_llm_client_class):
        # Mock PR description
        mock_pr_description = MagicMock()
        mock_pr_description.parsed = MagicMock(spec=CodePrReviewOutput.PrDescription)

        # Mock formatted response with comments without suggestions
        mock_comment = MagicMock(spec=CodePrReviewOutput.Comment)
        mock_comment.path = "test_file.py"
        mock_comment.line = 10
        mock_comment.body = "Test comment"
        mock_comment.start_line = 5
        mock_comment.suggestion = None  # No suggestion

        mock_formatted_response = MagicMock()
        mock_formatted_response.parsed = [mock_comment]

        # Configure LlmClient.generate_structured to return different responses on each call
        mock_llm_client_class.generate_structured.side_effect = [
            mock_pr_description,
            mock_formatted_response,
        ]

        # Call invoke
        result = self.component.invoke(self.request, mock_llm_client_class)

        # Verify suggestion was not added to comment body
        self.assertEqual(mock_comment.body, "Test comment")

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.comments, mock_formatted_response.parsed)
        self.assertEqual(result.description, mock_pr_description.parsed)
