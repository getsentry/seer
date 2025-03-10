import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.is_root_cause_obvious import (
    IsRootCauseObviousComponent,
    IsRootCauseObviousRequest,
)
from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPrompt,
    RootCauseAnalysisOutput,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.components.root_cause.prompts import RootCauseAnalysisPrompts
from seer.automation.autofix.prompts import format_repo_prompt
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class RootCauseAnalysisComponent(BaseComponent[RootCauseAnalysisRequest, RootCauseAnalysisOutput]):
    context: AutofixContext

    @observe(name="Root Cause Analysis")
    @ai_track(description="Root Cause Analysis")
    @inject
    def invoke(
        self, request: RootCauseAnalysisRequest, llm_client: LlmClient = injected
    ) -> RootCauseAnalysisOutput:
        is_obvious = (
            IsRootCauseObviousComponent(self.context).invoke(
                IsRootCauseObviousRequest(
                    event_details=request.event_details,
                    instruction=request.instruction,
                )
            )
            if not request.initial_memory
            else None
        )

        with BaseTools(self.context) as tools:
            state = self.context.state.get()

            readable_repos = state.readable_repos
            unreadable_repos = state.unreadable_repos

            agent = AutofixAgent(
                tools=(
                    tools.get_tools(can_access_repos=bool(readable_repos))
                    if not (is_obvious and is_obvious.is_root_cause_clear)
                    else None
                ),
                config=AgentConfig(
                    interactive=True,
                ),
                context=self.context,
                memory=request.initial_memory,
                name="Root Cause Analysis Agent",
            )

            repos_str = format_repo_prompt(readable_repos, unreadable_repos)

            try:
                has_tools = not (is_obvious and is_obvious.is_root_cause_clear)
                if has_tools:  # run context gatherer if not obvious
                    response = agent.run(
                        run_config=RunConfig(
                            model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                            system_prompt=RootCauseAnalysisPrompts.format_system_msg(
                                has_tools=True
                            ),
                            max_iterations=64,
                            memory_storage_key="root_cause_analysis",
                            run_name="Root Cause Discovery",
                            reasoning_effort="low",
                            temperature=1.0,
                            max_tokens=12000,  # Reduced from 32000
                        ),
                    )

                    if not response:
                        self.context.store_memory("root_cause_analysis", agent.memory)
                        return RootCauseAnalysisOutput(
                            causes=[],
                            termination_reason="Something went wrong when Autofix was running.",
                        )

                self.context.event_manager.add_log("Simulating profound thought...")

                # reason to propose final root cause
                agent.tools = []
                agent.memory = (
                    agent.memory
                    if has_tools
                    else (
                        [
                            Message(
                                role="user",
                                content=RootCauseAnalysisPrompts.format_default_msg(
                                    event=request.event_details.format_event(),
                                    summary=request.summary,
                                    code_map=request.profile,
                                    instruction=request.instruction,
                                    repos_str=repos_str,
                                    has_tools=False,
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
                        prompt=RootCauseAnalysisPrompts.root_cause_proposal_msg(),
                        system_prompt="You are an exceptional AI system that is amazing at analyzing bugs in codebases. Your job is to figure out the correct root cause of this issue.",
                        memory_storage_key="root_cause_analysis",
                        run_name="Root Cause Proposal",
                        temperature=1.0,
                        reasoning_effort="high",
                        max_tokens=12000,  # Reduced from 32000
                    )
                )

                if not response:
                    self.context.store_memory("root_cause_analysis", agent.memory)
                    return RootCauseAnalysisOutput(
                        causes=[],
                        termination_reason="Something went wrong when Autofix was running.",
                    )

                if "<NO_ROOT_CAUSES>" in response:
                    reason = response.split("<NO_ROOT_CAUSES>")[1].strip()
                    if "</NO_ROOT_CAUSES>" in reason:
                        reason = reason.split("</NO_ROOT_CAUSES>")[0].strip()
                    return RootCauseAnalysisOutput(causes=[], termination_reason=reason)

                self.context.event_manager.add_log(
                    "Arranging data in a way that looks intentional..."
                )

                formatted_response = llm_client.generate_structured(
                    messages=agent.memory,
                    prompt=RootCauseAnalysisPrompts.root_cause_formatter_msg(),
                    model=GeminiProvider.model("gemini-2.0-flash-001"),
                    response_format=MultipleRootCauseAnalysisOutputPrompt,
                    run_name="Root Cause Extraction & Formatting",
                    max_tokens=8192,
                )

                if not formatted_response or not getattr(formatted_response, "parsed", None):
                    return RootCauseAnalysisOutput(
                        causes=[],
                        termination_reason="Something went wrong when Autofix was running.",
                    )

                parsed = formatted_response.parsed
                cause = getattr(parsed, "cause", None)
                if not cause:
                    return RootCauseAnalysisOutput(
                        causes=[],
                        termination_reason="Something went wrong when Autofix was running.",
                    )

                cause_model = cause.to_model()
                cause_model.id = 0
                causes = [cause_model]
                return RootCauseAnalysisOutput(causes=causes, termination_reason=None)

            finally:
                with self.context.state.update() as cur:
                    cur.usage += agent.usage
