import logging

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.automation.autofix.autofix_agent import AutofixAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPrompt,
    RootCauseAnalysisOutput,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.components.root_cause.prompts import RootCauseAnalysisPrompts
from seer.automation.autofix.prompts import format_repo_prompt
from seer.automation.autofix.tools.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class RootCauseAnalysisComponent(BaseComponent[RootCauseAnalysisRequest, RootCauseAnalysisOutput]):
    context: AutofixContext

    @observe(name="Root Cause Analysis")
    @sentry_sdk.trace
    @inject
    def invoke(
        self,
        request: RootCauseAnalysisRequest,
        llm_client: LlmClient = injected,
        config: AppConfig = injected,
    ) -> RootCauseAnalysisOutput:
        with BaseTools(self.context) as tools:
            state = self.context.state.get()

            readable_repos = state.readable_repos
            unreadable_repos = state.unreadable_repos

            sentry_sdk.set_tag("is_rethinking", len(request.initial_memory) > 0)

            agent = AutofixAgent(
                tools=(tools.get_tools(can_access_repos=bool(readable_repos))),
                config=AgentConfig(
                    interactive=True,
                ),
                context=self.context,
                memory=request.initial_memory,
                name="Root Cause Analysis Agent",
            )

            repos_str = format_repo_prompt(readable_repos, unreadable_repos)

            try:
                response = agent.run(
                    run_config=RunConfig(
                        model=(
                            AnthropicProvider.model("claude-3-7-sonnet@20250219")
                            if config.SENTRY_REGION == "de"
                            else GeminiProvider.model("gemini-2.5-flash-preview-05-20")
                        ),
                        prompt=(
                            RootCauseAnalysisPrompts.format_default_msg(
                                event=request.event_details.format_event(),
                                summary=request.summary,
                                code_map=request.profile,
                                instruction=request.instruction,
                                trace_tree=request.trace_tree,
                                logs=request.logs,
                            )
                            if not request.initial_memory
                            else None
                        ),
                        system_prompt=RootCauseAnalysisPrompts.format_system_msg(
                            repos_str=repos_str, mode="context"
                        ),
                        max_iterations=64,
                        memory_storage_key="root_cause_analysis",
                        run_name="Root Cause Discovery",
                        temperature=0.0,
                        max_tokens=8192 if config.SENTRY_REGION == "de" else 32000,
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
                response = agent.run(
                    run_config=RunConfig(
                        model=AnthropicProvider.model("claude-3-7-sonnet@20250219"),
                        prompt=RootCauseAnalysisPrompts.root_cause_proposal_msg(),
                        system_prompt=RootCauseAnalysisPrompts.format_system_msg(
                            repos_str=repos_str, mode="reasoning"
                        ),
                        memory_storage_key="root_cause_analysis",
                        run_name="Root Cause Proposal",
                        temperature=1.0,
                        reasoning_effort="high",
                        max_tokens=32000,
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
                    model=(
                        GeminiProvider.model("gemini-2.0-flash-001")
                        if config.SENTRY_REGION == "de"
                        else GeminiProvider.model("gemini-2.5-flash-preview-05-20")
                    ),
                    response_format=MultipleRootCauseAnalysisOutputPrompt,
                    run_name="Root Cause Extraction & Formatting",
                    max_tokens=8192 if config.SENTRY_REGION == "de" else 32000,
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
