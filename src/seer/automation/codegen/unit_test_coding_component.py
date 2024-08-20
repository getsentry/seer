import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, ClaudeAgent
from seer.automation.autofix.autofix_context import AutofixContext
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

logger = logging.getLogger(__name__)


class UnitTestCodingComponent(BaseComponent[CodeUnitTestRequest, CodeUnitTestOutput]):
    context: CodegenContext

    @observe(name="Plan+UnitTest")
    @ai_track(description="Plan+UnitTest")
    def invoke(self, request: CodeUnitTestRequest) -> CodeUnitTestOutput | None:
        tools = BaseTools(self.context)

        agent = ClaudeAgent(
            tools=tools.get_tools(),
            config=AgentConfig(
                system_prompt=CodingUnitTestPrompts.format_system_msg(), max_iterations=24
            ),
        )

        final_response = agent.run(
            CodingUnitTestPrompts.format_unit_test_msg(diff_str=request.diff)
        )

        # with self.context.state.update() as cur:
        #     cur.usage += agent.usage

        if not final_response:
            return None

        plan_steps_content = extract_text_inside_tags(final_response, "plan_steps")

        coding_output = PlanStepsPromptXml.from_xml(
            f"<plan_steps>{escape_multi_xml(plan_steps_content, ['diff', 'description', 'commit_message'])}</plan_steps>"
        ).to_model()

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
                #     self._append_file_change(repo_client.repo_external_id, change)
            elif task.type == "file_delete":
                change = task_to_file_delete(task)
                file_changes.append(change)
                # self._append_file_change(
                #     repo_client.repo_external_id,
                #     change,
                # )
            elif task.type == "file_create":
                change = task_to_file_create(task)
                file_changes.append(change)
                # self._append_file_change(
                #     repo_client.repo_external_id,
                #     task_to_file_create(task),
                # )
            else:
                logger.warning(f"Unsupported task type: {task.type}")

        return CodeUnitTestOutput(diffs=file_changes)
