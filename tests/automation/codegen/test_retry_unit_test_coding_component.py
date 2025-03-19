from unittest.mock import MagicMock, patch
import pytest
from seer.automation.codegen.models import CodeUnitTestRequest
from seer.automation.models import FileChange
from seer.db import DbPrContextToUnitTestGenerationRunIdMapping
from seer.automation.codegen.retry_unittest_coding_component import RetryUnitTestCodingComponent


class TestRetryUnitTestCodingComponent:
    @pytest.fixture
    def component(self):
        mock_context = MagicMock()
        mock_context.get_memory.return_value = "prev_memory"
        mock_context.update_stored_memory = MagicMock()
        dummy_repo_client = MagicMock()
        dummy_repo_client.get_file_content.return_value = ("dummy file content", "utf-8")
        mock_context.get_repo_client.return_value = dummy_repo_client
        comp = RetryUnitTestCodingComponent(mock_context)
        comp.context = mock_context
        return comp

    @pytest.fixture
    def dummy_request(self):
        req = MagicMock(spec=CodeUnitTestRequest)
        req.codecov_client_params = {
            "repo_name": "repo1",
            "pullid": "123",
            "owner_username": "owner",
            "head_sha": "abc123",
        }
        return req

    @pytest.fixture
    def dummy_prev_context(self):
        ctx = MagicMock(spec=DbPrContextToUnitTestGenerationRunIdMapping)
        ctx.run_id = "prev_run_id"
        return ctx

    @patch("seer.automation.codegen.retry_unittest_coding_component.BaseTools")
    @patch("seer.automation.codegen.retry_unittest_coding_component.LlmAgent")
    def test_invoke_no_final_response(
        self, mock_llm_agent, mock_base_tools, component, dummy_request, dummy_prev_context
    ):
        dummy_tools = MagicMock()
        dummy_tools.get_tools.return_value = ["tool"]
        mock_base_tools.return_value.__enter__.return_value = dummy_tools
        mock_agent = MagicMock()
        mock_agent.run.return_value = None
        mock_llm_agent.return_value = mock_agent
        result = component.invoke(dummy_request, dummy_prev_context)
        assert result is None

    @patch(
        "seer.automation.codegen.retry_unittest_coding_component.extract_text_inside_tags",
        return_value="",
    )
    @patch("seer.automation.codegen.retry_unittest_coding_component.BaseTools")
    @patch("seer.automation.codegen.retry_unittest_coding_component.LlmAgent")
    def test_invoke_empty_plan_steps(
        self,
        mock_llm_agent,
        mock_base_tools,
        mock_extract,
        component,
        dummy_request,
        dummy_prev_context,
    ):
        dummy_tools = MagicMock()
        dummy_tools.get_tools.return_value = ["tool"]
        mock_base_tools.return_value.__enter__.return_value = dummy_tools
        mock_agent = MagicMock()
        mock_agent.run.return_value = "<plan_steps></plan_steps>"
        mock_llm_agent.return_value = mock_agent
        with pytest.raises(ValueError, match="Failed to extract plan_steps"):
            component.invoke(dummy_request, dummy_prev_context)

    @patch("seer.automation.codegen.retry_unittest_coding_component.PlanStepsPromptXml")
    @patch(
        "seer.automation.codegen.retry_unittest_coding_component.extract_text_inside_tags",
        return_value="non_empty",
    )
    @patch("seer.automation.codegen.retry_unittest_coding_component.BaseTools")
    @patch("seer.automation.codegen.retry_unittest_coding_component.LlmAgent")
    def test_invoke_no_tasks(
        self,
        mock_llm_agent,
        mock_base_tools,
        mock_extract,
        mock_plan_steps,
        component,
        dummy_request,
        dummy_prev_context,
    ):
        dummy_tools = MagicMock()
        dummy_tools.get_tools.return_value = ["tool"]
        mock_base_tools.return_value.__enter__.return_value = dummy_tools
        mock_agent = MagicMock()
        mock_agent.run.return_value = "<plan_steps>non_empty</plan_steps>"
        mock_llm_agent.return_value = mock_agent
        dummy_output = MagicMock()
        dummy_output.tasks = []
        mock_plan_steps.from_xml.return_value.to_model.return_value = dummy_output
        with pytest.raises(ValueError, match="No tasks found"):
            component.invoke(dummy_request, dummy_prev_context)

    @patch("seer.automation.codegen.retry_unittest_coding_component.task_to_file_create")
    @patch("seer.automation.codegen.retry_unittest_coding_component.task_to_file_delete")
    @patch("seer.automation.codegen.retry_unittest_coding_component.task_to_file_change")
    @patch("seer.automation.codegen.retry_unittest_coding_component.PlanStepsPromptXml")
    @patch(
        "seer.automation.codegen.retry_unittest_coding_component.extract_text_inside_tags",
        return_value="valid_plan",
    )
    @patch("seer.automation.codegen.retry_unittest_coding_component.BaseTools")
    @patch("seer.automation.codegen.retry_unittest_coding_component.LlmAgent")
    def test_invoke_successful(
        self,
        mock_llm_agent,
        mock_base_tools,
        mock_extract,
        mock_plan_steps,
        mock_task_change,
        mock_task_delete,
        mock_task_create,
        component,
        dummy_request,
        dummy_prev_context,
    ):
        dummy_tools = MagicMock()
        dummy_tools.get_tools.return_value = ["tool"]
        mock_base_tools.return_value.__enter__.return_value = dummy_tools
        mock_agent = MagicMock()
        mock_agent.run.return_value = "<plan_steps>valid_plan</plan_steps>"
        mock_llm_agent.return_value = mock_agent

        task_change = MagicMock()
        task_change.type = "file_change"
        task_change.file_path = "file_change.py"
        task_change.repo_name = "repo1"
        task_delete = MagicMock()
        task_delete.type = "file_delete"
        task_delete.file_path = "file_delete.py"
        task_delete.repo_name = "repo1"
        task_create = MagicMock()
        task_create.type = "file_create"
        task_create.file_path = "file_create.py"
        task_create.repo_name = "repo1"
        dummy_output = MagicMock()
        dummy_output.tasks = [task_change, task_delete, task_create]
        mock_plan_steps.from_xml.return_value.to_model.return_value = dummy_output

        mock_task_change.return_value = (
            [
                FileChange(
                    change_type="edit", path="file_change.py", diff="diff_change", commit_message=""
                )
            ],
            None,
        )
        mock_task_delete.return_value = FileChange(
            change_type="delete", path="file_delete.py", diff="diff_delete", commit_message=""
        )
        mock_task_create.return_value = FileChange(
            change_type="create", path="file_create.py", diff="diff_create", commit_message=""
        )

        repo_client = MagicMock()
        repo_client.get_file_content.return_value = ("some content", "utf-8")
        component.context.get_repo_client.return_value = repo_client

        result = component.invoke(dummy_request, dummy_prev_context)
        component.context.update_stored_memory.assert_called_once_with(
            "unit_test_memory", mock_agent.memory, dummy_prev_context.run_id
        )
        expected = [
            FileChange(
                change_type="edit", path="file_change.py", diff="diff_change", commit_message=""
            ),
            FileChange(
                change_type="delete", path="file_delete.py", diff="diff_delete", commit_message=""
            ),
            FileChange(
                change_type="create", path="file_create.py", diff="diff_create", commit_message=""
            ),
        ]
        assert result.diffs == expected
