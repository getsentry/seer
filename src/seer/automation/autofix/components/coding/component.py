import concurrent.futures
import json
import logging
import textwrap

import sentry_sdk
from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.automation.agent.models import Message, ToolCall
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.models import (
    CodeChangesPromptXml,
    CodeChangeXml,
    CodingOutput,
    CodingRequest,
    PlanTaskPromptXml,
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
from seer.configuration import AppConfig
from seer.dependency_injection import copy_modules_initializer, inject, injected

logger = logging.getLogger(__name__)


class CodingComponent(BaseComponent[CodingRequest, CodingOutput]):
    context: AutofixContext

    def _append_file_change(self, repo_external_id: str, file_change: FileChange):
        with self.context.state.update() as cur:
            cur.codebases[repo_external_id].file_changes.append(file_change)

    def _prefill_initial_memory(self, request: CodingRequest) -> list[Message]:
        memory: list[Message] = []

        relevant_files = []
        if isinstance(request.root_cause, RootCauseAnalysisItem):
            relevant_files_root_cause = list(
                {
                    (event.relevant_code_file.file_path, event.relevant_code_file.repo_name): {
                        "file_path": event.relevant_code_file.file_path,
                        "repo_name": event.relevant_code_file.repo_name,
                    }
                    for event in (
                        request.root_cause.root_cause_reproduction
                        if request.root_cause.root_cause_reproduction
                        else []
                    )
                    if event.relevant_code_file
                }.values()
            )
            relevant_files.extend(relevant_files_root_cause)

        if isinstance(request.solution, list):
            relevant_files_solution = list(
                {
                    (event.relevant_code_file.file_path, event.relevant_code_file.repo_name): {
                        "file_path": event.relevant_code_file.file_path,
                        "repo_name": event.relevant_code_file.repo_name,
                    }
                    for event in request.solution
                    if event.relevant_code_file
                }.values()
            )
            # Add only files that aren't already in the list
            for file in relevant_files_solution:
                if file not in relevant_files:
                    relevant_files.append(file)

        expanded_files_messages = []
        for i, file in enumerate(relevant_files):
            file_content = None
            try:
                file_content = (
                    self.context.get_file_contents(
                        path=file["file_path"], repo_name=file["repo_name"]
                    )
                    if file["file_path"]
                    else None
                )
            except Exception as e:
                logger.exception(f"Error getting file contents in memory prefill: {e}")
                file_content = None

            if file_content:
                agent_message = Message(
                    role="tool_use",
                    content=f"Expand document: {file['file_path']} in {file['repo_name']}",
                    tool_calls=[
                        ToolCall(
                            id=str(i),
                            function="expand_document",
                            args=json.dumps(
                                {"file_path": file["file_path"], "repo_name": file["repo_name"]}
                            ),
                        )
                    ],
                )
                user_message = Message(
                    role="tool",
                    content=file_content,
                    tool_call_id=str(i),
                    tool_call_function="expand_document",
                )
                memory.append(agent_message)
                memory.append(user_message)
                expanded_files_messages.append(agent_message)
                expanded_files_messages.append(user_message)

        with self.context.state.update() as cur:
            cur.steps[-1].initial_memory_length = len(expanded_files_messages) + 1

        return memory

    @observe(name="Apply Code")
    @ai_track(description="Apply Code")
    @inject
    def apply_code_suggestion_to_file(
        self,
        original_content: str | None,
        new_content: str,
        file_path: str,
        llm_client: LlmClient = injected,
    ) -> str | None:
        if original_content is None:
            # For new files, create a simple diff showing the entire file as added
            lines = new_content.splitlines()
            diff = f"--- /dev/null\n+++ b/{file_path}\n"
            diff += f"@@ -0,0 +1,{len(lines)} @@\n"
            for line in lines:
                diff += f"+{line}\n"
            return diff

        # For existing files, use LLM to merge changes
        system_prompt = """You are an coding assistant that helps merge code updates, ensuring every modification is fully integrated."""
        prompt = textwrap.dedent(
            """\
            You are an expert software engineer. You are given a code snippet and a proposed update to the code.
            You need to generate the exact unified diff that shows the changes to the code, but only specific portions. You do NOT need to include the entire file in the diff, but you should include up to 3 lines of unchanged lines before and after the changes for context. Please preserve indentation.

            Here is an example of a unified diff format:
            <example>
            --- a/path/to/file.py
            +++ b/path/to/file.py
            @@ -1,3 +1,3 @@
            x = 1
            y = 2
            -for i in range(10):
            -    print(x + y)
            +for i in range(20):
            +    print(x * y)
            </example>

            Here is the original code:
            <code>{original_content}</code>

            Here is the proposed update:
            <update>{new_content}</update>

            Provide the exact unified diff that covers the changes and no other text.
            """
        ).format(original_content=original_content, new_content=new_content)
        response = llm_client.generate_text(
            system_prompt=system_prompt,
            prompt=prompt,
            model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
        )
        if not response.message.content:
            return None
        return response.message.content + "\n"

    @observe(name="Is Obvious")
    @ai_track(description="Is Obvious")
    @inject
    def _is_obvious(
        self,
        request: CodingRequest,
        memory: list[Message],
        llm_client: LlmClient = injected,
    ) -> bool:
        if memory:

            class IsObviousOutput(BaseModel):
                need_to_search_codebase: bool

            output = llm_client.generate_structured(
                messages=memory,
                prompt=CodingPrompts.format_is_obvious_msg(
                    root_cause=request.root_cause,
                    original_instruction=request.original_instruction,
                    root_cause_extra_instruction=request.root_cause_extra_instruction,
                    custom_solution=request.solution if isinstance(request.solution, str) else None,
                    auto_solution=(
                        request.solution if isinstance(request.solution, list) else None
                    ),
                    mode=request.mode,
                ),
                model=GeminiProvider.model("gemini-2.0-flash-001"),
                response_format=IsObviousOutput,
            )

            return not output.parsed.need_to_search_codebase

        return False

    @observe(name="Feedback Check")
    @ai_track(description="Feedback Check")
    @inject
    def _is_feedback_obvious(self, memory: list[Message], llm_client: LlmClient = injected) -> bool:
        if memory:

            class NeedToSearchCodebaseOutput(BaseModel):
                need_to_search_codebase: bool

            output = llm_client.generate_structured(
                messages=memory,
                prompt="Given the above instruction, do you need to search the codebase for more context or have an immediate answer?",
                model=GeminiProvider.model("gemini-2.0-flash-001"),
                response_format=NeedToSearchCodebaseOutput,
            )

            with self.context.state.update() as cur:
                cur.usage += output.metadata.usage

            return not output.parsed.need_to_search_codebase

        return False

    @inject
    def _get_llm_client(self, llm_client: LlmClient = injected) -> LlmClient:
        return llm_client

    @inject
    def _get_app_config(self, app_config: AppConfig = injected) -> AppConfig:
        return app_config

    def _parse_code_changes_xml(
        self, response: str, agent: AutofixAgent
    ) -> CodeChangesPromptXml | None:
        """Try to parse code changes XML from response, with one retry attempt if parsing fails."""
        try:
            code_changes_content = extract_text_inside_tags(response, "code_changes")
            model = CodeChangesPromptXml.from_xml(
                f"<code_changes>{escape_multi_xml(code_changes_content, ['code', 'commit_message'])}</code_changes>"
            ).to_model()
            for task in model.tasks:
                task.repo_name = self.context.autocorrect_repo_name(task.repo_name)
            return model
        except Exception:
            # Try once to fix the XML format
            agent.config.interactive = False
            retry_response = agent.run(
                RunConfig(
                    prompt=CodingPrompts.format_xml_format_fix_msg(),
                    model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                    memory_storage_key="code",
                    run_name="XML Format Fix",
                ),
            )

            if not retry_response:
                return None

            try:
                code_changes_content = extract_text_inside_tags(retry_response, "code_changes")
                model = CodeChangesPromptXml.from_xml(
                    f"<code_changes>{escape_multi_xml(code_changes_content, ['code', 'commit_message'])}</code_changes>"
                ).to_model()
                for task in model.tasks:
                    task.repo_name = self.context.autocorrect_repo_name(task.repo_name)
                return model
            except Exception as e:
                sentry_sdk.capture_exception(e)
                return None

    def _fix_file_existence_errors(
        self, code_changes_output: CodeChangesPromptXml, agent: AutofixAgent
    ) -> CodeChangesPromptXml:
        """Check for missing or already existing files and try to fix the changes if needed."""
        missing_files_errors = []
        file_exist_errors = []
        correct_paths = []
        for task in code_changes_output.tasks:
            repo_client = self.context.get_repo_client(task.repo_name)
            file_content, _ = repo_client.get_file_content(task.file_path)
            if task.type == "file_change" and not file_content:
                missing_files_errors.append(task.file_path)
            elif task.type == "file_delete" and not file_content:
                missing_files_errors.append(task.file_path)
            elif task.type == "file_create" and file_content:
                file_exist_errors.append(task.file_path)
            else:
                correct_paths.append(task.file_path)

        if missing_files_errors or file_exist_errors:
            agent.config.interactive = False
            new_response = agent.run(
                RunConfig(
                    prompt=CodingPrompts.format_missing_msg(
                        missing_files_errors, file_exist_errors, correct_paths
                    ),
                    model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                    memory_storage_key="code",
                    run_name="Missing File Fix",
                ),
            )

            if new_response:
                new_output = self._parse_code_changes_xml(new_response, agent)
                if new_output:
                    return new_output

        return code_changes_output

    @observe(name="Code")
    @ai_track(description="Code")
    def invoke(self, request: CodingRequest) -> CodingOutput | None:
        with BaseTools(self.context) as tools:

            memory = request.initial_memory
            custom_solution = request.solution if isinstance(request.solution, str) else None
            auto_solution = request.solution if isinstance(request.solution, list) else None
            has_test = (
                any(step.timeline_item_type == "repro_test" for step in request.solution)
                if isinstance(request.solution, list)
                else False
            )

            is_obvious = False
            if not memory:
                memory = self._prefill_initial_memory(request=request)
                is_obvious = (
                    not has_test and request.mode == "fix" and self._is_obvious(request, memory)
                )  # coding is never obvious if we need to write a test
            else:
                is_obvious = self._is_feedback_obvious(memory)

            agent = AutofixAgent(
                tools=tools.get_tools() if not is_obvious else None,
                config=AgentConfig(interactive=True),
                memory=memory,
                context=self.context,
                name="Code",
            )

            if not request.initial_memory:
                agent.add_user_message(
                    CodingPrompts.format_fix_msg(
                        custom_solution=custom_solution,
                        auto_solution=auto_solution,
                        mode=request.mode,
                        has_test=has_test,
                        root_cause=request.root_cause,
                        event_details=request.event_details,
                    ),
                )

            response = agent.run(
                RunConfig(
                    system_prompt=CodingPrompts.format_system_msg(),
                    model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                    memory_storage_key="code",
                    run_name="Code",
                    max_iterations=64,
                ),
            )

            self.context.store_memory("code", agent.memory)

            with self.context.state.update() as cur:
                cur.usage += agent.usage

            if not response:
                raise ValueError("No response from coding agent")

            code_changes_output = self._parse_code_changes_xml(response, agent)
            if not code_changes_output:
                raise ValueError("No code changes output from coding agent")

            code_changes_output = self._fix_file_existence_errors(code_changes_output, agent)

            self.context.event_manager.add_log("Rewriting your unfortunate code...")
            tasks_with_diffs: list[PlanTaskPromptXml] = []

            @observe(name="Process Change Task")
            @ai_track(description="Process Change Task")
            def process_task(task: CodeChangeXml):
                repo_client = self.context.get_repo_client(task.repo_name)
                if task.type == "file_change":
                    file_content, _ = repo_client.get_file_content(task.file_path, autocorrect=True)
                    if not file_content:
                        logger.warning(f"Failed to get content for {task.file_path}")
                        return None
                    diff = self.apply_code_suggestion_to_file(
                        file_content, task.code, task.file_path
                    )
                    if not diff:
                        return None
                    task_with_diff = PlanTaskPromptXml(
                        file_path=task.file_path,
                        repo_name=task.repo_name or repo_client.repo_full_name,
                        type="file_change",
                        diff=diff,
                        commit_message=task.commit_message,
                        description=f"Change file {task.file_path}",
                    )
                    changes, _ = task_to_file_change(task_with_diff, file_content)
                    updates = [(repo_client.repo_external_id, change) for change in changes]
                    return task_with_diff, updates

                elif task.type == "file_delete":
                    task_with_diff = PlanTaskPromptXml(
                        file_path=task.file_path,
                        repo_name=task.repo_name or repo_client.repo_full_name,
                        type="file_delete",
                        commit_message=task.commit_message,
                        diff="",
                        description=f"Delete file {task.file_path}",
                    )
                    update = (repo_client.repo_external_id, task_to_file_delete(task_with_diff))
                    return task_with_diff, [update]

                elif task.type == "file_create":
                    diff = self.apply_code_suggestion_to_file(None, task.code, task.file_path)
                    if not diff:
                        return None
                    task_with_diff = PlanTaskPromptXml(
                        file_path=task.file_path,
                        repo_name=task.repo_name or repo_client.repo_full_name,
                        type="file_create",
                        diff=diff,
                        commit_message=task.commit_message,
                        description=f"Create file {task.file_path}",
                    )
                    update = (repo_client.repo_external_id, task_to_file_create(task_with_diff))
                    return task_with_diff, [update]

                else:
                    logger.warning(f"Unsupported task type: {task.type}")
                    return None

            # apply change tasks in parallel
            with concurrent.futures.ThreadPoolExecutor(
                initializer=copy_modules_initializer()
            ) as executor:
                trace_id = langfuse_context.get_current_trace_id()
                observation_id = langfuse_context.get_current_observation_id()

                futures = [
                    executor.submit(
                        process_task,
                        task,
                        langfuse_parent_trace_id=trace_id,  # type: ignore
                        langfuse_parent_observation_id=observation_id,  # type: ignore
                    )
                    for task in code_changes_output.tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        task_with_diff, updates = result
                        tasks_with_diffs.append(task_with_diff)
                        for repo_external_id, file_change in updates:
                            self._append_file_change(repo_external_id, file_change)
                    else:
                        sentry_sdk.capture_message("Failed to apply code changes.")

            return CodingOutput(tasks=tasks_with_diffs)
