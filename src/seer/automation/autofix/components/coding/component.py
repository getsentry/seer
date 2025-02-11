import concurrent.futures
import difflib
import logging
import textwrap

import sentry_sdk
from langfuse.decorators import observe
from openai import BadRequestError as OpenAiBadRequestError
from openai import LengthFinishReasonError as OpenAiLengthFinishReasonError
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import (
    AnthropicProvider,
    GeminiProvider,
    LlmClient,
    OpenAiProvider,
)
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.models import (
    CodeChangesPromptXml,
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
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class CodingComponent(BaseComponent[CodingRequest, CodingOutput]):
    context: AutofixContext

    def _append_file_change(self, repo_external_id: str, file_change: FileChange):
        with self.context.state.update() as cur:
            cur.codebases[repo_external_id].file_changes.append(file_change)

    def _prefill_initial_memory(self) -> list[Message]:
        memory: list[Message] = []

        solution_memory = self.context.get_memory("solution")
        if solution_memory:
            memory.extend(solution_memory)

        with self.context.state.update() as cur:
            cur.steps[-1].initial_memory_length = len(memory) + 1

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
            """
            Merge all changes from the <update> snippet into the <code> below.
            - Preserve the code's structure, order, comments, and indentation exactly.
            - Output only the updated code, enclosed within <updated_code> and </updated_code> tags.
            - Do not include any other text, explanations, placeholders, ellipses, or code fences.

            <code>{original_content}</code>

            <update>{new_content}</update>

            Provide the complete updated code.
            """
        ).format(original_content=original_content, new_content=new_content)

        # use predicted output for faster response
        predicted_output = f"<updated_code>{original_content}</updated_code>"
        try:
            output = llm_client.generate_text(
                system_prompt=system_prompt,
                messages=[Message(role="user", content=prompt)],
                model=OpenAiProvider.model("gpt-4o-mini"),
                predicted_output=predicted_output,
            )
        except (
            OpenAiBadRequestError,
            OpenAiLengthFinishReasonError,
        ) as e:  # too much content, fallback to model with bigger input/output limit
            sentry_sdk.capture_message(
                f"Failed to apply code suggestion to file with gpt-4o-mini, falling back to o3-mini. Error message: {str(e)}"
            )
            try:
                output = llm_client.generate_text(
                    system_prompt=system_prompt,
                    messages=[Message(role="user", content=prompt)],
                    model=OpenAiProvider.model("o3-mini"),
                )
            except Exception as e2:
                sentry_sdk.capture_exception(e2)
                return None

        text = output.message.content
        updated_content = extract_text_inside_tags(text, "updated_code")

        # Generate unified diff between original_content and updated_content
        original_lines = original_content.splitlines()
        updated_lines = updated_content.splitlines()
        diff = "\n".join(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
        return diff

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
                    summary=request.summary,
                    event_details=request.event_details,
                    root_cause=request.root_cause,
                    original_instruction=request.original_instruction,
                    root_cause_extra_instruction=request.root_cause_extra_instruction,
                    custom_solution=request.solution if isinstance(request.solution, str) else None,
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

    @observe(name="Code")
    @ai_track(description="Code")
    def invoke(self, request: CodingRequest) -> CodingOutput | None:
        with BaseTools(self.context) as tools:

            memory = request.initial_memory

            is_obvious = False
            if not memory:
                memory = self._prefill_initial_memory()
                is_obvious = self._is_obvious(request, memory)
            else:
                is_obvious = self._is_feedback_obvious(memory)

            agent = AutofixAgent(
                tools=tools.get_tools() if not is_obvious else None,
                config=AgentConfig(interactive=True),
                memory=memory,
                context=self.context,
                name="Code",
            )

            custom_solution = request.solution if isinstance(request.solution, str) else None

            if not request.initial_memory:
                agent.add_user_message(
                    CodingPrompts.format_fix_msg(
                        has_tools=not is_obvious,
                        custom_solution=custom_solution,
                        mode=request.mode,
                    ),
                )

            response = agent.run(
                RunConfig(
                    system_prompt=CodingPrompts.format_system_msg(has_tools=not is_obvious),
                    model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                    memory_storage_key="code",
                    run_name="Code",
                ),
            )

            self.context.store_memory("code", agent.memory)

            with self.context.state.update() as cur:
                cur.usage += agent.usage

            if not response:
                return None

            code_changes_content = extract_text_inside_tags(response, "code_changes")
            code_changes_output = CodeChangesPromptXml.from_xml(
                f"<code_changes>{escape_multi_xml(code_changes_content, ['code', 'commit_message'])}</code_changes>"
            ).to_model()

            # We only do this once, if it still errors, we just let it go
            missing_files_errors = []
            file_exist_errors = []
            for task in code_changes_output.tasks:
                repo_client = self.context.get_repo_client(task.repo_name)
                file_content, _ = repo_client.get_file_content(task.file_path, autocorrect=True)
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
                        memory_storage_key="code",
                        run_name="Missing File Fix",
                    ),
                )

                if new_response and "<code_changes>" in new_response:
                    code_changes_output = CodeChangesPromptXml.from_xml(
                        f"<code_changes>{escape_multi_xml(extract_text_inside_tags(new_response, 'code_changes'), ['code', 'commit_message'])}</code_changes>"
                    ).to_model()

        self.context.event_manager.add_log("Applying changes...")
        tasks_with_diffs: list[PlanTaskPromptXml] = []

        # Resolve LlmClient once in the main thread
        resolved_llm_client = self._get_llm_client()

        def process_task(task, llm_client):
            repo_client = self.context.get_repo_client(task.repo_name)
            if task.type == "file_change":
                file_content, _ = repo_client.get_file_content(task.file_path, autocorrect=True)
                if not file_content:
                    logger.warning(f"Failed to get content for {task.file_path}")
                    return None
                diff = self.apply_code_suggestion_to_file(
                    file_content, task.code, task.file_path, llm_client=llm_client
                )
                if not diff:
                    return None
                task_with_diff = PlanTaskPromptXml(
                    file_path=task.file_path,
                    repo_name=task.repo_name,
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
                    repo_name=task.repo_name,
                    type="file_delete",
                    commit_message=task.commit_message,
                    diff="",
                    description=f"Delete file {task.file_path}",
                )
                update = (repo_client.repo_external_id, task_to_file_delete(task_with_diff))
                return task_with_diff, [update]

            elif task.type == "file_create":
                diff = self.apply_code_suggestion_to_file(
                    None, task.code, task.file_path, llm_client=llm_client
                )
                if not diff:
                    return None
                task_with_diff = PlanTaskPromptXml(
                    file_path=task.file_path,
                    repo_name=task.repo_name,
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_task, task, resolved_llm_client)
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
