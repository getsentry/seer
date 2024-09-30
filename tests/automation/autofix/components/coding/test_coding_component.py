from unittest.mock import MagicMock, patch

import pytest

from seer.automation.agent.client import ClaudeClient
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.component import CodingComponent
from seer.automation.autofix.components.coding.models import (
    FileMissingObj,
    FuzzyDiffChunk,
    PlanTaskPromptXml,
)
from seer.automation.models import FileChange
from seer.dependency_injection import Module


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
    def mock_claude_client(self):
        return MagicMock(spec=ClaudeClient)

    def test_handle_missing_file_changes_success(self, component, mock_claude_client):
        missing_changes = {
            "file1.py": FileMissingObj(
                file_path="file1.py",
                file_content="original content",
                diff_chunks=[
                    FuzzyDiffChunk(
                        header="chunk1",
                        original_chunk="original content",
                        new_chunk="new content",
                        diff_content="diff content",
                    )
                ],
                task=PlanTaskPromptXml(
                    description="description",
                    commit_message="commit message",
                    file_path="file1.py",
                    repo_name="test_repo",
                    type="file_change",
                    diff="old diff",
                ),
            )
        }

        mock_claude_client.completion.return_value = (
            MagicMock(content="<corrected_diffs>new diff</corrected_diffs>"),
            MagicMock(),
        )

        mock_repo_client = MagicMock()
        mock_repo_client.repo_external_id = "test_repo_id"
        component.context.get_repo_client.return_value = mock_repo_client

        with patch(
            "seer.automation.autofix.components.coding.component.task_to_file_change"
        ) as mock_task_to_file_change:
            mock_task_to_file_change.return_value = (
                [
                    FileChange(
                        path="file1.py",
                        change_type="edit",
                        reference_snippet="old",
                        new_snippet="new",
                    )
                ],
                [],
            )

            module = Module()
            module.constant(ClaudeClient, mock_claude_client)
            with module:
                component._handle_missing_file_changes(missing_changes)

        component.context.state.update.assert_called()
        mock_claude_client.completion.assert_called_once()
        mock_task_to_file_change.assert_called_once()
        component._append_file_change.assert_called_once_with(
            "test_repo_id",
            FileChange(
                path="file1.py", change_type="edit", reference_snippet="old", new_snippet="new"
            ),
        )

    def test_handle_missing_file_changes_no_content(self, component, mock_claude_client):
        missing_changes = {
            "file1.py": FileMissingObj(
                file_path="file1.py",
                file_content="original content",
                diff_chunks=[
                    FuzzyDiffChunk(
                        header="chunk1",
                        original_chunk="original content",
                        new_chunk="new content",
                        diff_content="diff content",
                    )
                ],
                task=PlanTaskPromptXml(
                    description="description",
                    commit_message="commit message",
                    file_path="file1.py",
                    repo_name="test_repo",
                    type="file_change",
                    diff="old diff",
                ),
            )
        }

        mock_claude_client.completion.return_value = (MagicMock(content=None), MagicMock())

        module = Module()
        module.constant(ClaudeClient, mock_claude_client)
        with module:
            component._handle_missing_file_changes(missing_changes)

        component.context.state.update.assert_called()
        mock_claude_client.completion.assert_called_once()
        component._append_file_change.assert_not_called()

    def test_handle_missing_file_changes_with_remaining_missing_changes(
        self, component, mock_claude_client
    ):
        missing_changes = {
            "file1.py": FileMissingObj(
                file_path="file1.py",
                file_content="original content",
                diff_chunks=[
                    FuzzyDiffChunk(
                        header="chunk1",
                        original_chunk="original content",
                        new_chunk="new content",
                        diff_content="diff content",
                    )
                ],
                task=PlanTaskPromptXml(
                    description="description",
                    commit_message="commit message",
                    file_path="file1.py",
                    repo_name="test_repo",
                    type="file_change",
                    diff="old diff",
                ),
            )
        }

        mock_claude_client.completion.return_value = (
            MagicMock(content="<corrected_diffs>new diff</corrected_diffs>"),
            MagicMock(),
        )

        mock_repo_client = MagicMock()
        mock_repo_client.repo_external_id = "test_repo_id"
        component.context.get_repo_client.return_value = mock_repo_client

        with (
            patch(
                "seer.automation.autofix.components.coding.component.task_to_file_change"
            ) as mock_task_to_file_change,
            patch(
                "seer.automation.autofix.components.coding.component.append_langfuse_observation_metadata"
            ) as mock_append_metadata,
            patch(
                "seer.automation.autofix.components.coding.component.append_langfuse_trace_tags"
            ) as mock_append_tags,
        ):

            mock_task_to_file_change.return_value = (
                [
                    FileChange(
                        path="file1.py",
                        change_type="edit",
                        reference_snippet="old",
                        new_snippet="new",
                    )
                ],
                ["remaining_chunk"],
            )

            module = Module()
            module.constant(ClaudeClient, mock_claude_client)
            with module:
                component._handle_missing_file_changes(missing_changes)

        component.context.state.update.assert_called()
        mock_claude_client.completion.assert_called_once()
        mock_task_to_file_change.assert_called_once()
        component._append_file_change.assert_called_once_with(
            "test_repo_id",
            FileChange(
                path="file1.py", change_type="edit", reference_snippet="old", new_snippet="new"
            ),
        )
        mock_append_metadata.assert_called_once_with({"missing_changes_count": 1})
        mock_append_tags.assert_called_once_with(["missing_changes_count:1"])
