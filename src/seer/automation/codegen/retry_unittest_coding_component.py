import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from integrations.codecov.codecov_client import CodecovClient
from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient
from seer.automation.autofix.tools import BaseTools
from seer.automation.autofix.components.coding.models import PlanStepsPromptXml
from seer.automation.autofix.components.coding.utils import (
    task_to_file_change,
    task_to_file_create,
    task_to_file_delete,
)
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodeUnitTestOutput, CodeUnitTestRequest
from seer.automation.codegen.prompts import CodingUnitTestPrompts, RetryUnitTestPrompts
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags
from seer.db import DbPrContextToUnitTestGenerationRunIdMapping
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class RetryUnitTestCodingComponent(BaseComponent[CodeUnitTestRequest, CodeUnitTestOutput]):
    context: CodegenContext

    @observe(name="Retry unit tests")
    @ai_track(description="Retry unit test generation")
    @inject
    def invoke(
        self,
        request: CodeUnitTestRequest,
        previous_run_context: DbPrContextToUnitTestGenerationRunIdMapping,
        llm_client: LlmClient = injected,
    ) -> CodeUnitTestOutput | None:
        with BaseTools(self.context, repo_client_type=RepoClientType.CODECOV_PR_REVIEW) as tools:
            previous_run_memory = self.context.get_memory(
                "UnitTestRunMemory", past_run_id=previous_run_context.run_id
            )

            agent = LlmAgent(
                tools=tools.get_tools(),
                config=AgentConfig(interactive=False),
                memory=previous_run_memory,
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

            final_response = agent.run(
                run_config=RunConfig(
                    prompt=RetryUnitTestPrompts.format_continue_unit_tests_prompt(
                        code_coverage_info=code_coverage_data,
                        test_result_info=test_result_data,
                    ),
                    system_prompt=CodingUnitTestPrompts.format_system_msg(),
                    model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                    run_name="Retry Unit Tests",
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
                repo_client = self.context.get_repo_client(
                    task.repo_name, type=RepoClientType.CODECOV_PR_REVIEW
                )
                repo_client.base_commit_sha = codecov_client_params["head_sha"]
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

            self.context.update_stored_memory(
                "unit_test_memory", agent.memory, previous_run_context.run_id
            )
            return CodeUnitTestOutput(diffs=file_changes)
