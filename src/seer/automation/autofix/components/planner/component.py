import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, GptAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.planner.models import (
    PlanningOutput,
    PlanningOutputPromptXml,
    PlanningRequest,
    RootCausePlanTaskPromptXml,
)
from seer.automation.autofix.components.planner.prompts import PlanningPrompts
from seer.automation.autofix.components.planner.utils import extract_diff_chunks
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.tools import BaseTools
from seer.automation.autofix.utils import find_original_snippet
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, remove_cdata
from seer.langfuse import append_langfuse_trace_tags

logger = logging.getLogger(__name__)


class PlanningComponent(BaseComponent[PlanningRequest, PlanningOutput]):
    context: AutofixContext

    @observe(name="Planning")
    @ai_track(description="Planning")
    def invoke(self, request: PlanningRequest) -> PlanningOutput | None:
        tools = BaseTools(self.context)

        agent = GptAgent(
            tools=tools.get_tools(),
            config=AgentConfig(system_prompt=PlanningPrompts.format_system_msg()),
        )

        task_str = (
            RootCausePlanTaskPromptXml.from_root_cause(request.root_cause_and_fix).to_prompt_str()
            if isinstance(request.root_cause_and_fix, RootCauseAnalysisItem)
            else request.root_cause_and_fix
        )

        response = agent.run(
            PlanningPrompts.format_default_msg(
                event=request.event_details,
                task_str=task_str,
                instruction=request.instruction,
            ),
            context=self.context,
        )

        if not response:
            return None

        with self.context.state.update() as cur:
            cur.usage += agent.usage

        planning_output = PlanningOutputPromptXml.from_xml(
            f"<planning_output>{remove_cdata(escape_multi_xml(response, ['thoughts', 'diff']))}</planning_output>"
        ).to_model()

        for task in planning_output.tasks:
            repo_client = self.context.get_repo_client(task.repo_name)
            if task.type == "file_change":
                diff_chunks = extract_diff_chunks(task.diff)

                file_content = repo_client.get_file_content(task.file_path)

                if not file_content:
                    raise ValueError(
                        f"File {task.file_path} not found in repository {task.repo_name}"
                    )

                for chunk in diff_chunks:
                    result = find_original_snippet(
                        chunk.original_chunk,
                        file_content,
                        threshold=0.75,
                        initial_line_threshold=0.95,
                    )

                    if result:
                        original_snippet = result[0]

                        with self.context.state.update() as cur:
                            cur.codebases[repo_client.repo_external_id].file_changes.append(
                                FileChange(
                                    change_type="edit",
                                    path=task.file_path,
                                    reference_snippet=original_snippet,
                                    new_snippet=chunk.new_chunk,
                                ),
                            )
                    else:
                        logger.info(f"Original snippet not found in file {task.file_path}")
                        append_langfuse_trace_tags(["skipped_file_change:snippet_not_found"])
            elif task.type == "file_delete":
                with self.context.state.update() as cur:
                    cur.codebases[repo_client.repo_external_id].file_changes.append(
                        FileChange(
                            change_type="delete",
                            path=task.file_path,
                        ),
                    )
            elif task.type == "file_create":
                diff_chunks = extract_diff_chunks(task.diff)

                if len(diff_chunks) != 1:
                    raise ValueError(
                        f"Expected exactly one diff chunk for file creation, got {len(diff_chunks)}"
                    )

                chunk = diff_chunks[0]

                with self.context.state.update() as cur:
                    cur.codebases[repo_client.repo_external_id].file_changes.append(
                        FileChange(
                            change_type="create",
                            path=task.file_path,
                            new_snippet=chunk.new_chunk,
                        ),
                    )
            else:
                logger.warning(f"Unsupported task type: {task.type}")

        return planning_output
