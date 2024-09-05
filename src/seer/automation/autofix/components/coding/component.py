import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, GptAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.models import (
    CodingOutput,
    CodingRequest,
    PlanStepsPromptXml,
    RootCausePlanTaskPromptXml,
)
from seer.automation.autofix.components.coding.prompts import CodingPrompts
from seer.automation.autofix.components.coding.utils import (
    task_to_file_change,
    task_to_file_create,
    task_to_file_delete,
)
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags

logger = logging.getLogger(__name__)


class CodingComponent(BaseComponent[CodingRequest, CodingOutput]):
    context: AutofixContext

    def _append_file_change(self, repo_external_id: str, file_change: FileChange):
        with self.context.state.update() as cur:
            cur.codebases[repo_external_id].file_changes.append(file_change)

    @observe(name="Plan+Code")
    @ai_track(description="Plan+Code")
    def invoke(self, request: CodingRequest) -> CodingOutput | None:
        tools = BaseTools(self.context)

        agent = GptAgent(
            tools=tools.get_tools(),
            config=AgentConfig(system_prompt=CodingPrompts.format_system_msg()),
        )

        task_str = (
            RootCausePlanTaskPromptXml.from_root_cause(request.root_cause_and_fix).to_prompt_str()
            if isinstance(request.root_cause_and_fix, RootCauseAnalysisItem)
            else request.root_cause_and_fix
        )

        state = self.context.state.get()

        response = agent.run(
            CodingPrompts.format_fix_discovery_msg(
                event=request.event_details.format_event(),
                task_str=task_str,
                summary=request.summary,
                repo_names=[repo.full_name for repo in state.request.repos],
                instruction=request.instruction,
            ),
            context=self.context,
        )

        prev_usage = agent.usage
        with self.context.state.update() as cur:
            cur.usage += agent.usage

        if not response:
            return None

        final_response = agent.run(CodingPrompts.format_fix_msg())

        with self.context.state.update() as cur:
            cur.usage += agent.usage - prev_usage

        if not final_response:
            return None

        plan_steps_content = extract_text_inside_tags(final_response, "plan_steps")

        coding_output = PlanStepsPromptXml.from_xml(
            f"<plan_steps>{escape_multi_xml(plan_steps_content, ['diff', 'description', 'commit_message'])}</plan_steps>"
        ).to_model()

        for task in coding_output.tasks:
            repo_client = self.context.get_repo_client(task.repo_name)
            if task.type == "file_change":
                file_content = repo_client.get_file_content(task.file_path)

                if not file_content:
                    logger.warning(f"Failed to get content for {task.file_path}")
                    continue

                for change in task_to_file_change(task, file_content):
                    self._append_file_change(repo_client.repo_external_id, change)
            elif task.type == "file_delete":
                self._append_file_change(
                    repo_client.repo_external_id,
                    task_to_file_delete(task),
                )
            elif task.type == "file_create":
                self._append_file_change(
                    repo_client.repo_external_id,
                    task_to_file_create(task),
                )
            else:
                logger.warning(f"Unsupported task type: {task.type}")

        return coding_output
