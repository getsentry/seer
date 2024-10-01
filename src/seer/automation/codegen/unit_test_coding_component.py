import logging

from langfuse.decorators import langfuse_context, observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, ClaudeAgent, LlmAgent
from seer.automation.autofix.components.coding.models import PlanStepsPromptXml
from seer.automation.autofix.components.coding.utils import (
    task_to_file_change,
    task_to_file_create,
    task_to_file_delete,
)
from seer.automation.autofix.tools import BaseTools
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodeUnitTestOutput, CodeUnitTestRequest
from seer.automation.codegen.prompts import CodingUnitTestPrompts
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags
from integrations.codecov.codecov_client import CodecovClient

logger = logging.getLogger(__name__)


class UnitTestCodingComponent(BaseComponent[CodeUnitTestRequest, CodeUnitTestOutput]):
    context: CodegenContext

    @observe(name="Analyze existing test design")
    @ai_track(description="AnalyzeTestDesign")
    def _get_test_design_summary(self, agent: LlmAgent, prompt: str):
        return agent.run(prompt)

    @observe(name="Create plan")
    @ai_track(description="GetTestPlan")
    def _get_plan(self, agent: LlmAgent, prompt: str) -> str:
        return agent.run(prompt)

    @observe(name="Generate test tasks")
    @ai_track(description="GenerateTests")
    def _generate_tests(self, agent: LlmAgent, prompt: str) -> str:
        return agent.run(prompt=prompt)

    def invoke(self, request: CodeUnitTestRequest) -> CodeUnitTestOutput | None:
        langfuse_context.update_current_trace(user_id="ram")
        tools = BaseTools(self.context)

        agent = ClaudeAgent(
            tools=tools.get_tools(),
            config=AgentConfig(
                system_prompt=CodingUnitTestPrompts.format_system_msg(), max_iterations=24
            ),
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

        existing_test_design_response = self._get_test_design_summary(
            agent=agent,
            prompt=CodingUnitTestPrompts.format_find_unit_test_pattern_step_msg(
                diff_str=request.diff
            ),
        )

        self._get_plan(
            agent=agent,
            prompt=CodingUnitTestPrompts.format_plan_step_msg(
                diff_str=request.diff,
                has_coverage_info=code_coverage_data,
                has_test_result_info=test_result_data,
            ),
        )

        final_response = self._generate_tests(
            agent=agent,
            prompt=CodingUnitTestPrompts.format_unit_test_msg(
                diff_str=request.diff, test_design_hint=existing_test_design_response
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
        for task in coding_output.tasks:
            repo_client = self.context.get_repo_client(task.repo_name)
            if task.type == "file_change":
                file_content = repo_client.get_file_content(task.file_path)
                if not file_content:
                    logger.warning(f"Failed to get content for {task.file_path}")
                    continue

                for change in task_to_file_change(task, file_content):
                    file_changes.append(change)
            elif task.type == "file_delete":
                change = task_to_file_delete(task)
                file_changes.append(change)
            elif task.type == "file_create":
                change = task_to_file_create(task)
                file_changes.append(change)
            else:
                logger.warning(f"Unsupported task type: {task.type}")

        return CodeUnitTestOutput(diffs=file_changes)
