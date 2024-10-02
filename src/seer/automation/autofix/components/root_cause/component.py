import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, GptAgent
from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPrompt,
    RootCauseAnalysisOutput,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.components.root_cause.prompts import RootCauseAnalysisPrompts
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.automation.utils import extract_parsed_model
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class RootCauseAnalysisComponent(BaseComponent[RootCauseAnalysisRequest, RootCauseAnalysisOutput]):
    context: AutofixContext

    @observe(name="Root Cause Analysis")
    @ai_track(description="Root Cause Analysis")
    @inject
    def invoke(
        self, request: RootCauseAnalysisRequest, gpt_client: GptClient = injected
    ) -> RootCauseAnalysisOutput | None:
        with BaseTools(self.context) as tools:
            agent = GptAgent(
                tools=tools.get_tools(),
                config=AgentConfig(
                    system_prompt=RootCauseAnalysisPrompts.format_system_msg(),
                    max_iterations=24,
                    interactive=True,
                ),
                name="root_cause_analysis",
                memory=request.initial_memory,
            )

            state = self.context.state.get()

            try:
                response = agent.run(
                    (
                        RootCauseAnalysisPrompts.format_default_msg(
                            event=request.event_details.format_event(),
                            summary=request.summary,
                            instruction=request.instruction,
                            repo_names=[repo.full_name for repo in state.request.repos],
                        )
                        if not request.initial_memory
                        else None
                    ),
                    context=self.context,
                )

                if not response:
                    self.context.store_memory("root_cause_analysis", agent.memory)
                    return None

                if "<NO_ROOT_CAUSES>" in response:
                    return None

                # Ask for reproduction
                self.context.event_manager.add_log("Thinking about how to reproduce the issue...")
                agent.run(
                    RootCauseAnalysisPrompts.reproduction_prompt_msg(),
                )

                self.context.store_memory("root_cause_analysis", agent.memory)

                self.context.event_manager.add_log("Cleaning up my findings...")
                response = gpt_client.openai_client.beta.chat.completions.parse(
                    messages=[
                        message.to_message()
                        for message in gpt_client.clean_tool_call_assistant_messages(agent.memory)
                    ]
                    + [
                        Message(
                            role="user",
                            content=RootCauseAnalysisPrompts.root_cause_formatter_msg(),
                        ).to_message(),
                    ],
                    model="gpt-4o-2024-08-06",
                    response_format=MultipleRootCauseAnalysisOutputPrompt,
                )

                parsed = extract_parsed_model(response)

                # Assign the ids to be the numerical indices of the causes and relevant code context
                cause_model = parsed.cause.to_model()
                cause_model.id = 0
                if cause_model.code_context:
                    for j, snippet in enumerate(cause_model.code_context):
                        snippet.id = j

                causes = [cause_model]
                return RootCauseAnalysisOutput(causes=causes)
            finally:
                with self.context.state.update() as cur:
                    cur.usage += agent.usage
