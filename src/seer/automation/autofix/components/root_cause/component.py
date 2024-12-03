import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient, OpenAiProvider
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
from seer.automation.autofix.tools import BaseTools
from seer.automation.autofix.utils import find_original_snippet
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
                IsRootCauseObviousRequest(event_details=request.event_details)
            )
            if not request.initial_memory
            else None
        )

        with BaseTools(self.context) as tools:
            agent = AutofixAgent(
                tools=(
                    tools.get_tools()
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

            state = self.context.state.get()

            try:
                response = agent.run(
                    run_config=RunConfig(
                        model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                        prompt=(
                            RootCauseAnalysisPrompts.format_default_msg(
                                event=request.event_details.format_event(),
                                summary=request.summary,
                                instruction=request.instruction,
                                repo_names=[repo.full_name for repo in state.request.repos],
                            )
                            if not request.initial_memory
                            else None
                        ),
                        system_prompt=RootCauseAnalysisPrompts.format_system_msg(
                            has_tools=not (is_obvious and is_obvious.is_root_cause_clear)
                        ),
                        max_iterations=24,
                        memory_storage_key="root_cause_analysis",
                        run_name="Root Cause Discovery",
                    ),
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

                # Ask for reproduction (NOTE: disabled due to speed; should be relocated to a different step in the future)
                # self.context.event_manager.add_log("Thinking about how to reproduce the issue...")
                # response = agent.run(
                #     run_config=RunConfig(
                #         model=OpenAiProvider.model("gpt-4o-2024-08-06"),
                #         prompt=RootCauseAnalysisPrompts.reproduction_prompt_msg(),
                #         run_name="Root Cause Reproduction & Unit Test",
                #     )
                # )
                # if not response:
                #     self.context.store_memory("root_cause_analysis", agent.memory)
                #     return RootCauseAnalysisOutput(
                #         causes=[],
                #         termination_reason="Something went wrong when Autofix was trying to figure out how to reproduce the issue.",
                #     )

                self.context.event_manager.add_log("Cleaning up the findings...")

                formatted_response = llm_client.generate_structured(
                    messages=agent.memory,
                    prompt=RootCauseAnalysisPrompts.root_cause_formatter_msg(),
                    model=OpenAiProvider.model("gpt-4o-2024-08-06"),
                    response_format=MultipleRootCauseAnalysisOutputPrompt,
                    run_name="Root Cause Extraction & Formatting",
                )

                # Assign the ids to be the numerical indices of the causes and relevant code context
                cause_model = formatted_response.parsed.cause.to_model()
                cause_model.id = 0
                if cause_model.code_context:
                    for i, item in enumerate(cause_model.code_context):
                        item.id = i
                        # Find line range for the snippet
                        if item.snippet.file_path and item.snippet.snippet:
                            try:
                                file_contents = self.context.get_file_contents(
                                    item.snippet.file_path,
                                )
                                if file_contents:
                                    found_snippet = find_original_snippet(
                                        item.snippet.snippet,
                                        file_contents,
                                        threshold=0.95,
                                    )
                                    if found_snippet:
                                        _, start_line, end_line = found_snippet
                                        item.snippet.start_line = start_line
                                        item.snippet.end_line = end_line
                            except Exception as e:
                                logger.warning(
                                    f"Failed to find line numbers for code context snippet: {e}"
                                )

                causes = [cause_model]
                return RootCauseAnalysisOutput(causes=causes, termination_reason=None)
            finally:
                with self.context.state.update() as cur:
                    cur.usage += agent.usage
