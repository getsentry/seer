import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from integrations.codecov.codecov_client import CodecovClient
from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient
from seer.automation.autofix.components.coding.models import PlanStepsPromptXml
from seer.automation.autofix.components.coding.utils import (
    task_to_file_change,
    task_to_file_create,
    task_to_file_delete,
)
from seer.automation.autofix.tools.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodeUnitTestOutput, CodeUnitTestRequest
from seer.automation.codegen.prompts import CodingUnitTestPrompts
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class UnitTestCodingComponent(BaseComponent[CodeUnitTestRequest, CodeUnitTestOutput]):
    context: CodegenContext

    @observe(name="Generate unit tests")
    @ai_track(description="Generate unit tests")
    @inject
    def invoke(
        self,
        request: CodeUnitTestRequest,
        is_codecov_request: bool,
        llm_client: LlmClient = injected,
    ) -> CodeUnitTestOutput | None:
        with BaseTools(self.context, repo_client_type=RepoClientType.CODECOV_UNIT_TEST) as tools:
            agent = LlmAgent(
                tools=tools.get_tools(),
                config=AgentConfig(interactive=False),
            )

            codecov_client_params = request.codecov_client_params

            code_coverage_data = CodecovClient.fetch_coverage(
                repo_name=codecov_client_params["repo_name"],
                pullid=codecov_client_params["pullid"],
                owner_username=codecov_client_params["owner_username"],
            )

            test_result_data = CodecovClient.fetch_test_results_for_commit(
                repo_name=codecov_client_params["repo_name"],
                owner_username=codecov_client_params["owner_username"],
                latest_commit_sha=codecov_client_params["head_sha"],
            )

            existing_test_design_response = llm_client.generate_text(
                model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                prompt=CodingUnitTestPrompts.format_find_unit_test_pattern_step_msg(
                    diff_str=request.diff
                ),
            )

            llm_client.generate_text(
                model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                prompt=CodingUnitTestPrompts.format_plan_step_msg(
                    diff_str=request.diff,
                    has_coverage_info=code_coverage_data,
                    has_test_result_info=test_result_data,
                ),
            )

            final_response = agent.run(
                run_config=RunConfig(
                    prompt=CodingUnitTestPrompts.format_unit_test_msg(
                        diff_str=request.diff,
                        test_design_hint=existing_test_design_response,
                    ),
                    system_prompt=CodingUnitTestPrompts.format_system_msg(),
                    model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                    run_name="Generate Unit Tests",
                    max_iterations=64,
                ),
            )

            if not final_response:
                return None
            plan_steps_content = extract_text_inside_tags(final_response, "plan_steps")

            if len(plan_steps_content) == 0:
                raise ValueError("Failed to extract plan_steps from the planning step of LLM")

            coding_output = PlanStepsPromptXml.from_xml(
                f"<plan_steps>{escape_multi_xml(plan_steps_content, ['diff', 'description', 'commit_message'])}</plan_steps>"
            ).to_model()

        if not coding_output.tasks:
            raise ValueError("No tasks found in coding output")
        file_changes: list[FileChange] = []
        client_type = (
            RepoClientType.CODECOV_PR_REVIEW
            if is_codecov_request
            else RepoClientType.CODECOV_UNIT_TEST
        )
        for task in coding_output.tasks:
            repo_client = self.context.get_repo_client(repo_name=task.repo_name, type=client_type)
            if task.type == "file_change":
                file_content, _ = repo_client.get_file_content(task.file_path)
                if not file_content:
                    logger.warning(f"Failed to get content for {task.file_path}")
                    continue

                changes, _ = task_to_file_change(task, file_content)
                file_changes += changes
            elif task.type == "file_delete":
                change = task_to_file_delete(task)
                file_changes.append(change)
            elif task.type == "file_create":
                change = task_to_file_create(task)
                file_changes.append(change)
            else:
                logger.warning(f"Unsupported task type: {task.type}")

        self.context.store_memory("unit_test_memory", agent.memory)
        return CodeUnitTestOutput(diffs=file_changes)
