import json
import logging

from langfuse.decorators import observe
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.automation.agent.models import Message, ToolCall
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.models import CodingOutput, CodingRequest
from seer.automation.autofix.components.coding.prompts import CodingPrompts
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.prompts import format_repo_prompt
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class CodingComponent(BaseComponent[CodingRequest, CodingOutput]):
    context: AutofixContext

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
                file_content, corrected_path = (
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

    @observe(name="Code")
    @ai_track(description="Code")
    def invoke(self, request: CodingRequest) -> None:
        with BaseTools(self.context) as tools:
            memory = request.initial_memory
            custom_solution = request.solution if isinstance(request.solution, str) else None
            auto_solution = request.solution if isinstance(request.solution, list) else None
            has_test = (
                any(step.timeline_item_type == "repro_test" for step in request.solution)
                if isinstance(request.solution, list)
                else False
            )

            if not memory:
                memory = self._prefill_initial_memory(request=request)

            agent = AutofixAgent(
                tools=tools.get_tools(include_claude_tools=True),
                config=AgentConfig(interactive=True),
                memory=memory,
                context=self.context,
                name="Code",
            )

            if not request.initial_memory:
                state = self.context.state.get()
                agent.add_user_message(
                    CodingPrompts.format_fix_msg(
                        custom_solution=custom_solution,
                        auto_solution=auto_solution,
                        mode=request.mode,
                        has_test=has_test,
                        root_cause=request.root_cause,
                        event_details=request.event_details,
                        repos_str=format_repo_prompt(
                            readable_repos=state.readable_repos,
                            unreadable_repos=state.unreadable_repos,
                            is_using_claude_tools=True,
                        ),
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
