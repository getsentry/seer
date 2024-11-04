import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.models import (
    CodingOutput,
    CodingRequest,
    FileMissingObj,
    PlanStepsPromptXml,
    RootCausePlanTaskPromptXml,
)
from seer.automation.autofix.components.coding.prompts import CodingPrompts
from seer.automation.autofix.components.coding.utils import (
    task_to_file_change,
    task_to_file_create,
    task_to_file_delete,
)
from seer.automation.autofix.components.is_fix_obvious import (
    IsFixObviousComponent,
    IsFixObviousRequest,
)
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags
from seer.dependency_injection import inject, injected
from seer.langfuse import append_langfuse_observation_metadata, append_langfuse_trace_tags

logger = logging.getLogger(__name__)


class CodingComponent(BaseComponent[CodingRequest, CodingOutput]):
    context: AutofixContext

    def _append_file_change(self, repo_external_id: str, file_change: FileChange):
        with self.context.state.update() as cur:
            cur.codebases[repo_external_id].file_changes.append(file_change)

    @observe(name="Incorrect diff fixer")
    @ai_track(description="Incorrect diff fixer")
    @inject
    def _handle_missing_file_changes(
        self,
        missing_changes_by_file: dict[str, FileMissingObj],
        llm_client: LlmClient = injected,
    ):
        for file_path, file_missing_obj in missing_changes_by_file.items():
            new_response = llm_client.generate_text(
                model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                prompt=CodingPrompts.format_incorrect_diff_fixer(
                    file_path,
                    file_missing_obj.diff_chunks,
                    file_missing_obj.file_content,
                ),
                temperature=0.0,
                run_name="Incorrect Diff Fixer",
            )

            with self.context.state.update() as cur:
                cur.usage += new_response.metadata.usage

            if not new_response.message.content:
                continue

            corrected_diffs = extract_text_inside_tags(
                new_response.message.content, "corrected_diffs"
            )
            new_task = file_missing_obj.task.model_copy(update={"diff": corrected_diffs})

            changes, missing_changes = task_to_file_change(new_task, file_missing_obj.file_content)

            # If there are any more missing changes, we just ignore at this point
            missing_changes_count = len(missing_changes)
            append_langfuse_observation_metadata(
                {
                    "missing_changes_count": missing_changes_count,
                }
            )
            if missing_changes_count > 0:
                append_langfuse_trace_tags([f"missing_changes_count:{missing_changes_count}"])

            repo_client = self.context.get_repo_client(new_task.repo_name)
            for change in changes:
                self._append_file_change(repo_client.repo_external_id, change)

    @observe(name="Plan+Code")
    @ai_track(description="Plan+Code")
    def invoke(self, request: CodingRequest) -> CodingOutput | None:
        with BaseTools(self.context) as tools:
            agent = AutofixAgent(
                tools=tools.get_tools(),
                config=AgentConfig(interactive=True),
                memory=request.initial_memory,
                context=self.context,
                name="Plan+Code",
            )

            task_str = (
                RootCausePlanTaskPromptXml.from_root_cause(
                    request.root_cause_and_fix
                ).to_prompt_str()
                if isinstance(request.root_cause_and_fix, RootCauseAnalysisItem)
                else request.root_cause_and_fix
            )

            has_tools = True

            if (
                isinstance(request.root_cause_and_fix, RootCauseAnalysisItem)
                and not request.initial_memory
            ):
                expanded_files_messages = []
                relevant_files = [
                    {"file_path": context.snippet.file_path, "repo_name": context.snippet.repo_name}
                    for context in (
                        request.root_cause_and_fix.code_context
                        if request.root_cause_and_fix.code_context
                        else []
                    )
                    if context.snippet
                ]
                for file in relevant_files:
                    file_content = (
                        self.context.get_file_contents(
                            path=file["file_path"], repo_name=file["repo_name"]
                        )
                        if file["file_path"]
                        else None
                    )
                    if file_content:
                        agent_message = Message(
                            role="assistant",
                            content=f"Expand document: {file['file_path']} in {file['repo_name']}",
                        )
                        user_message = Message(
                            role="user",
                            content=file_content,
                        )
                        agent.memory.append(agent_message)
                        agent.memory.append(user_message)
                        expanded_files_messages.append(agent_message)
                        expanded_files_messages.append(user_message)

                with self.context.state.update() as cur:
                    cur.steps[-1].initial_memory_length = len(expanded_files_messages) + 1

                if expanded_files_messages and not request.initial_memory:
                    is_obvious = IsFixObviousComponent(self.context).invoke(
                        IsFixObviousRequest(
                            event_details=request.event_details,
                            task_str=task_str,
                            fix_instruction=request.fix_instruction,
                            memory=expanded_files_messages,
                        )
                    )
                    if is_obvious and is_obvious.is_fix_clear:
                        agent.tools = []
                        has_tools = False

            state = self.context.state.get()

            response = agent.run(
                run_config=RunConfig(
                    prompt=(
                        CodingPrompts.format_fix_discovery_msg(
                            event=request.event_details.format_event(),
                            task_str=task_str,
                            summary=request.summary,
                            repo_names=[repo.full_name for repo in state.request.repos],
                            instruction=request.instruction,
                            fix_instruction=request.fix_instruction,
                            has_tools=has_tools,
                        )
                        if not request.initial_memory
                        else None
                    ),
                    system_prompt=CodingPrompts.format_system_msg(has_tools=has_tools),
                    model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                    memory_storage_key="plan_and_code",
                    run_name="Plan",
                ),
            )

            prev_usage = agent.usage
            with self.context.state.update() as cur:
                cur.usage += agent.usage

            if not response:
                self.context.store_memory("plan_and_code", agent.memory)
                return None

            final_response = agent.run(
                RunConfig(
                    prompt=CodingPrompts.format_fix_msg(),
                    model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                    memory_storage_key="plan_and_code",
                    run_name="Code",
                    has_tools=has_tools,
                ),
            )

            self.context.store_memory("plan_and_code", agent.memory)

            with self.context.state.update() as cur:
                cur.usage += agent.usage - prev_usage

            if not final_response:
                return None

            plan_steps_content = extract_text_inside_tags(final_response, "plan_steps")

            coding_output = PlanStepsPromptXml.from_xml(
                f"<plan_steps>{escape_multi_xml(plan_steps_content, ['diff', 'description', 'commit_message'])}</plan_steps>"
            ).to_model()

            # We only do this once, if it still errors, we just let it go
            missing_files_errors = []
            file_exist_errors = []
            for task in coding_output.tasks:
                repo_client = self.context.get_repo_client(task.repo_name)
                file_content = repo_client.get_file_content(task.file_path)
                if task.type == "file_change" and not file_content:
                    missing_files_errors.append(task.file_path)
                elif task.type == "file_delete" and not file_content:
                    missing_files_errors.append(task.file_path)
                elif task.type == "file_create" and file_content:
                    file_exist_errors.append(task.file_path)

            if missing_files_errors or file_exist_errors:
                agent.config.interactive = False
                new_response = agent.run(
                    RunConfig(
                        prompt=CodingPrompts.format_missing_msg(
                            missing_files_errors, file_exist_errors
                        ),
                        model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                        memory_storage_key="plan_and_code",
                        run_name="Missing File Fix",
                    ),
                )

                if new_response and "<plan_steps>" in new_response:
                    coding_output = PlanStepsPromptXml.from_xml(
                        f"<plan_steps>{escape_multi_xml(extract_text_inside_tags(new_response, 'plan_steps'), ['diff', 'description', 'commit_message'])}</plan_steps>"
                    ).to_model()

        missing_changes_by_file: dict[str, FileMissingObj] = dict()

        for task in coding_output.tasks:
            repo_client = self.context.get_repo_client(task.repo_name)
            if task.type == "file_change":
                file_content = repo_client.get_file_content(task.file_path)

                if not file_content:
                    logger.warning(f"Failed to get content for {task.file_path}")
                    continue

                changes, missing_changes = task_to_file_change(task, file_content)

                for change in changes:
                    self._append_file_change(repo_client.repo_external_id, change)

                if missing_changes:
                    missing_changes_by_file[task.file_path] = FileMissingObj(
                        file_path=task.file_path,
                        file_content=file_content,
                        diff_chunks=missing_changes,
                        task=task,
                    )
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

        if missing_changes_by_file:
            self._handle_missing_file_changes(missing_changes_by_file)

        return coding_output
