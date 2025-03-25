import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.solution.models import SolutionOutput, SolutionRequest
from seer.automation.autofix.components.solution.prompts import SolutionPrompts
from seer.automation.autofix.prompts import format_repo_prompt
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class SolutionComponent(BaseComponent[SolutionRequest, SolutionOutput]):
    context: AutofixContext

    def _prefill_initial_memory(self, request: SolutionRequest) -> list[Message]:
        memory: list[Message] = []

        root_cause_memory = self.context.get_memory("root_cause_analysis")
        if root_cause_memory:
            memory.extend(root_cause_memory)

        with self.context.state.update() as cur:
            cur.steps[-1].initial_memory_length = len(memory) + 1

        return memory

    @observe(name="Solution")
    @ai_track(description="Solution")
    @inject
    def invoke(
        self, request: SolutionRequest, llm_client: LlmClient = injected
    ) -> SolutionOutput | None:

        with BaseTools(self.context) as tools:
            memory = request.initial_memory

            state = self.context.state.get()

            readable_repos = state.readable_repos
            unreadable_repos = state.unreadable_repos

            if not memory:
                memory = self._prefill_initial_memory(request)

            has_tools = bool(readable_repos)
            repos_str = format_repo_prompt(readable_repos, unreadable_repos)

            agent = AutofixAgent(
                tools=tools.get_tools() if has_tools else None,
                config=AgentConfig(interactive=True),
                memory=memory,
                context=self.context,
                name="Solution",
            )

            if not request.initial_memory:
                agent.add_user_message(
                    SolutionPrompts.format_default_msg(
                        event=request.event_details.format_event(),
                        root_cause=request.root_cause_and_fix,
                        summary=request.summary,
                        repos_str=repos_str,
                        original_instruction=request.original_instruction,
                        code_map=request.profile,
                        trace_tree=(
                            request.trace_tree
                            if (
                                state.request.invoking_user
                                and state.request.invoking_user.id == 3283725
                            )
                            else None
                        ),  # TODO temporary guard for Rohan (@roaga) to test in prod
                    ),
                )

            try:
                if has_tools:  # run context gatherer
                    response = agent.run(
                        run_config=RunConfig(
                            model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                            system_prompt=SolutionPrompts.format_system_msg(),
                            memory_storage_key="solution",
                            run_name="Solution Discovery",
                            max_iterations=64,
                            temperature=1.0,
                            reasoning_effort="medium",
                            max_tokens=32000,
                        ),
                    )

                    if not response:
                        self.context.store_memory("solution", agent.memory)
                        return None

                self.context.event_manager.add_log("Being artificially intelligent...")

                # reason to propose final solution
                agent.tools = []
                agent.memory = (
                    agent.memory
                    if has_tools
                    else (
                        [
                            Message(
                                role="user",
                                content=SolutionPrompts.format_default_msg(
                                    event=request.event_details.format_event(),
                                    root_cause=request.root_cause_and_fix,
                                    summary=request.summary,
                                    repos_str=repos_str,
                                    original_instruction=request.original_instruction,
                                    code_map=request.profile,
                                    trace_tree=request.trace_tree,
                                ),
                            )
                        ]
                        if not request.initial_memory
                        else request.initial_memory
                    )
                )

                response = agent.run(
                    run_config=RunConfig(
                        model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                        memory_storage_key="solution",
                        run_name="Solution Proposal",
                        temperature=1.0,
                        reasoning_effort="high",
                        prompt=SolutionPrompts.solution_proposal_msg(),
                        system_prompt="You are an exceptional AI system that is amazing at fixing bugs in codebases. Your job is to figure out the correct and most effective solution to fix this issue.",
                        max_tokens=32000,
                    )
                )

                if not response:
                    self.context.store_memory("solution", agent.memory)
                    return None

                self.context.event_manager.add_log("Formatting for human consumption...")

                formatted_response = llm_client.generate_structured(
                    messages=agent.memory,
                    prompt=SolutionPrompts.solution_formatter_msg(request.root_cause_and_fix),
                    model=GeminiProvider.model("gemini-2.0-flash-001"),
                    response_format=SolutionOutput,
                    run_name="Solution Extraction & Formatting",
                    max_tokens=8192,
                )

                if not formatted_response or not formatted_response.parsed:
                    return None

                return formatted_response.parsed

            finally:
                with self.context.state.update() as cur:
                    cur.usage += agent.usage
