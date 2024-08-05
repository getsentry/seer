import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, GptAgent
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisOutput,
    RootCauseAnalysisOutputPromptXml,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.components.root_cause.prompts import RootCauseAnalysisPrompts
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.automation.utils import escape_multi_xml

logger = logging.getLogger(__name__)


class RootCauseAnalysisComponent(BaseComponent[RootCauseAnalysisRequest, RootCauseAnalysisOutput]):
    context: AutofixContext

    @observe(name="Root Cause Analysis")
    @ai_track(description="Root Cause Analysis")
    def invoke(self, request: RootCauseAnalysisRequest) -> RootCauseAnalysisOutput | None:
        tools = BaseTools(self.context)

        agent = GptAgent(
            tools=tools.get_tools(),
            config=AgentConfig(
                system_prompt=RootCauseAnalysisPrompts.format_system_msg(), max_iterations=24
            ),
        )

        response = agent.run(
            RootCauseAnalysisPrompts.format_default_msg(
                event=request.event_details,
                instruction=request.instruction,
            ),
            context=self.context,
        )

        original_usage = agent.usage
        with self.context.state.update() as cur:
            cur.usage += agent.usage

        if not response:
            logger.warning("Root Cause Analysis agent did not return a valid response")
            return None

        if "<NO_ROOT_CAUSES>" in response:
            return None

        agent.config.model = "gpt-4o-mini-2024-07-18"
        formatter_response = agent.run(RootCauseAnalysisPrompts.root_cause_formatter_msg())

        with self.context.state.update() as cur:
            cur.usage += agent.usage - original_usage

        if not formatter_response:
            logger.warning("Root Cause Analysis formatter did not return a valid response")
            return None

        xml_response = RootCauseAnalysisOutputPromptXml.from_xml(
            f"<root>{escape_multi_xml(formatter_response, ['thoughts', 'title', 'description', 'code'])}</root>"
        )

        # Assign the ids to be the numerical indices of the causes and relevant code context
        causes = []
        for i, cause in enumerate(xml_response.potential_root_causes.causes):
            cause_model = cause.to_model()
            cause_model.id = i

            if cause_model.code_context:
                for j, snippet in enumerate(cause_model.code_context):
                    snippet.id = j

            causes.append(cause_model)

        return RootCauseAnalysisOutput(causes=causes)
