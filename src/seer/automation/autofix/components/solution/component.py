import json
import logging

from langfuse.decorators import observe
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient, OpenAiProvider
from seer.automation.agent.models import Message, ToolCall
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.components.solution.models import SolutionOutput, SolutionRequest
from seer.automation.autofix.components.solution.prompts import SolutionPrompts
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class SolutionComponent(BaseComponent[SolutionRequest, SolutionOutput]):
    context: AutofixContext

    def _prefill_initial_memory(self, request: SolutionRequest) -> list[Message]:
        memory: list[Message] = []

        if isinstance(request.root_cause_and_fix, RootCauseAnalysisItem):
            expanded_files_messages = []
            relevant_files = list(
                {
                    (event.relevant_code_file.file_path, event.relevant_code_file.repo_name): {
                        "file_path": event.relevant_code_file.file_path,
                        "repo_name": event.relevant_code_file.repo_name,
                    }
                    for event in (
                        request.root_cause_and_fix.root_cause_reproduction
                        if request.root_cause_and_fix.root_cause_reproduction
                        else []
                    )
                    if event.relevant_code_file
                }.values()
            )

            for i, file in enumerate(relevant_files):
                file_content = (
                    self.context.get_file_contents(
                        path=file["file_path"], repo_name=file["repo_name"]
                    )
                    if file["file_path"]
                    else None
                )
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
        request: SolutionRequest,
        memory: list[Message],
        llm_client: LlmClient = injected,
    ) -> bool:
        if memory:

            class IsObviousOutput(BaseModel):
                need_to_search_codebase: bool

            output = llm_client.generate_structured(
                messages=memory,
                prompt=SolutionPrompts.format_is_obvious_msg(
                    summary=request.summary,
                    event_details=request.event_details,
                    root_cause=request.root_cause_and_fix,
                    original_instruction=request.original_instruction,
                ),
                model=OpenAiProvider.model("gpt-4o-mini"),
                response_format=IsObviousOutput,
            )

            return not output.parsed.need_to_search_codebase

        return False

    @observe(name="Solution")
    @ai_track(description="Solution")
    @inject
    def invoke(
        self, request: SolutionRequest, llm_client: LlmClient = injected
    ) -> SolutionOutput | None:

        with BaseTools(self.context) as tools:
            memory = request.initial_memory

            is_obvious = False
            if not memory:
                memory = self._prefill_initial_memory(request)
                is_obvious = self._is_obvious(request, memory)

            agent = AutofixAgent(
                tools=tools.get_tools() if not is_obvious else None,
                config=AgentConfig(interactive=True),
                memory=memory,
                context=self.context,
                name="Solution",
            )

            state = self.context.state.get()
            if not request.initial_memory:
                agent.memory.insert(
                    0,
                    Message(
                        role="user",
                        content=SolutionPrompts.format_default_msg(
                            event=request.event_details.format_event(),
                            root_cause=request.root_cause_and_fix,
                            summary=request.summary,
                            repo_names=[repo.full_name for repo in state.request.repos],
                            original_instruction=request.original_instruction,
                            code_map=request.profile,
                            has_tools=not is_obvious,
                        ),
                    ),
                )

            response = agent.run(
                run_config=RunConfig(
                    model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                    system_prompt=SolutionPrompts.format_system_msg(has_tools=not is_obvious),
                    memory_storage_key="solution",
                    run_name="Solution",
                )
            )

            with self.context.state.update() as cur:
                cur.usage += agent.usage

            if not response:
                self.context.store_memory("solution", agent.memory)
                return None

            self.context.event_manager.add_log("Cleaning up the findings...")

            formatted_response = llm_client.generate_structured(
                messages=agent.memory,
                prompt=SolutionPrompts.solution_formatter_msg(request.root_cause_and_fix),
                model=OpenAiProvider.model("gpt-4o-mini"),
                response_format=SolutionOutput,
                run_name="Solution Extraction & Formatting",
                max_tokens=4096,
            )

            if not formatted_response or not formatted_response.parsed:
                return None

            return formatted_response.parsed
