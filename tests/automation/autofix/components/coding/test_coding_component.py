from unittest.mock import MagicMock, patch

import pytest
from johen import generate

from seer.automation.agent.client import LlmClient
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.component import CodingComponent
from seer.automation.autofix.components.coding.models import CodeChangeXml, CodingRequest
from seer.automation.autofix.components.root_cause.models import (
    RelevantCodeFile,
    RootCauseAnalysisItem,
)
from seer.automation.models import EventDetails


class TestCodingComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock(spec=AutofixContext)
        mock_context.state = MagicMock()
        mock_context.skip_loading_codebase = True
        component = CodingComponent(mock_context)
        component._append_file_change = MagicMock()
        return component

    @pytest.fixture
    def mock_llm_client(self):
        return MagicMock(spec=LlmClient)

    @patch(
        "seer.automation.autofix.components.coding.component.extract_text_inside_tags",
        side_effect=["", "First code changes", "Updated code changes"],
    )
    @patch("seer.automation.autofix.components.coding.component.CodeChangesPromptXml")
    def test_invoke_with_missing_and_existing_files(
        self, mock_code_changes_prompt_xml, mock_extract_text, component, mock_llm_client
    ):
        # Setup
        request = next(generate(CodingRequest))
        request.initial_memory = []

        mock_agent = MagicMock()
        mock_agent.run.side_effect = [
            "Initial response",
            "<code_changes>First code changes</code_changes>",
            "<code_changes>Updated code changes</code_changes>",
        ]
        mock_agent.usage = MagicMock()

        mock_repo_client = MagicMock()
        component.context.get_repo_client.return_value = mock_repo_client
        component.context.event_manager = MagicMock()
        component._is_obvious = MagicMock(return_value=False)

        # Simulate file content for different scenarios
        def mock_get_file_content(file_path, autocorrect=False):
            if file_path in ["missing_file1.py", "missing_file2.py"]:
                return None, "utf-8"
            elif file_path == "existing_file.py":
                return "Existing content", "utf-8"
            return "def foo():\n    return 'Hello'", "utf-8"

        mock_repo_client.get_file_content.side_effect = mock_get_file_content

        mock_coding_output = MagicMock()
        mock_coding_output.tasks = [
            CodeChangeXml(
                type="file_change",
                file_path="missing_file1.py",
                repo_name="repo1",
                code="return 'Hello, World!",
                commit_message="",
            ),
            CodeChangeXml(
                type="file_delete",
                file_path="missing_file2.py",
                repo_name="repo1",
                code="return 'Hello, World!",
                commit_message="",
            ),
            CodeChangeXml(
                type="file_create",
                file_path="existing_file.py",
                repo_name="repo1",
                code="return 'Hello, World!",
                commit_message="",
            ),
        ]

        mock_code_changes_prompt_xml.from_xml.return_value.to_model.return_value = (
            mock_coding_output
        )

        # Execute
        with patch(
            "seer.automation.autofix.components.coding.component.AutofixAgent",
            return_value=mock_agent,
        ):
            result = component.invoke(request)

        # Assert
        assert mock_agent.run.call_count == 2
        assert "missing_file1.py" in mock_agent.run.call_args_list[1][0][0].prompt
        assert "missing_file2.py" in mock_agent.run.call_args_list[1][0][0].prompt
        assert "existing_file.py" in mock_agent.run.call_args_list[1][0][0].prompt

        # Ensure the result is as expected
        for task in result.tasks:
            assert task.type in ["file_change", "file_delete", "file_create"]
            assert task.file_path in ["missing_file1.py", "missing_file2.py", "existing_file.py"]
            assert task.repo_name == "repo1"
            assert task.commit_message == ""

    def test_invoke_with_root_cause_analysis_non_obvious_fix(self, component):
        # Setup
        mock_request = MagicMock()
        mock_request.root_cause_and_fix = MagicMock(spec=RootCauseAnalysisItem)
        mock_request.root_cause_and_fix.root_cause_reproduction = [
            MagicMock(
                title="Test title",
                code_snippet_and_analysis="Test description",
                timeline_item_type="internal_code",
                relevant_code_file=RelevantCodeFile(file_path="test.py", repo_name="test-repo"),
                is_most_important_event=True,
            )
        ]
        mock_request.event_details = EventDetails(
            title="Test Error",
        )
        mock_request.fix_instruction = "Fix the code"
        mock_request.initial_memory = None

        mock_repo_client = MagicMock()
        mock_repo_client.get_file_content.return_value = "test content"
        component.context.get_file_contents.return_value = "test content"
        component.context.get_repo_client.return_value = mock_repo_client

        component._is_obvious = MagicMock(return_value=False)
        component.context.event_manager = MagicMock()

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = "<code_changes>test plan</code_changes>"
        mock_agent.usage = MagicMock()
        mock_tools = ["tool1", "tool2"]
        mock_agent.tools = mock_tools

        # Execute
        with (
            patch(
                "seer.automation.autofix.components.coding.component.AutofixAgent",
                return_value=mock_agent,
            ),
            patch(
                "seer.automation.autofix.components.coding.component.CodeChangesPromptXml"
            ) as mock_code_changes,
        ):
            mock_code_changes.from_xml.return_value.to_model.return_value = MagicMock()
            component.invoke(mock_request)

        # Assert
        component._is_obvious.assert_called_once()
        # Verify agent keeps its tools when fix is not obvious
        assert mock_agent.tools == mock_tools
