from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, GptAgent
from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisOutput,
    RootCauseAnalysisOutputPromptXml,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.components.root_cause.prompts import RootCauseAnalysisPrompts
from seer.automation.autofix.tools import BaseTools
from seer.automation.autofix.utils import autofix_logger
from seer.automation.component import BaseComponent
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags


class RootCauseAnalysisComponent(BaseComponent[RootCauseAnalysisRequest, RootCauseAnalysisOutput]):
    context: AutofixContext

    @observe(name="Root Cause Analysis")
    @ai_track(description="Root Cause Analysis")
    def invoke(self, request: RootCauseAnalysisRequest) -> RootCauseAnalysisOutput | None:
        tools = BaseTools(self.context)

        agent = GptAgent(
            tools=tools.get_tools(),
            memory=[],
            config=AgentConfig(system_prompt=RootCauseAnalysisPrompts.format_system_msg()),
        )

        response = agent.run(
            RootCauseAnalysisPrompts.format_default_msg(
                event=request.event_details,
                instruction=request.instruction,
            )
        )

        with self.context.state.update() as cur:
            cur.usage += agent.usage

        if not response:
            autofix_logger.warning("Root Cause Analysis agent did not return a valid response")
            return None

        if "<NO_ROOT_CAUSES>" in response:
            return None

        extracted_response = extract_text_inside_tags(response, "potential_root_causes")

        formatter_response, formatter_usage = GptClient().completion(
            [
                Message(
                    role="user",
                    content=RootCauseAnalysisPrompts.root_cause_formatter_msg(extracted_response),
                )
            ]
        )

        with self.context.state.update() as cur:
            cur.usage += formatter_usage

        if not formatter_response.content:
            autofix_logger.warning("Root Cause Analysis formatter did not return a valid response")
            return None

        xml_response = RootCauseAnalysisOutputPromptXml.from_xml(
            f"<root>{escape_multi_xml(formatter_response.content, ['thoughts', 'snippet', 'title', 'description'])}</root>"
        )

        # Assign the ids to be the numerical indices of the causes and suggested fixes
        causes = []
        for i, cause in enumerate(xml_response.potential_root_causes.causes):
            cause_model = cause.to_model()
            cause_model.id = i

            if cause_model.suggested_fixes:
                for j, suggested_fix in enumerate(cause_model.suggested_fixes):
                    suggested_fix.id = j

            causes.append(cause_model)

        return RootCauseAnalysisOutput(causes=causes)
