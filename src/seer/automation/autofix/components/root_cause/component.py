import logging

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
from seer.automation.component import BaseComponent
from seer.automation.utils import escape_multi_xml, extract_text_inside_tags

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

        # preprocess event information
        event_details_response, _ = GptClient().completion(
            messages=[
                Message(
                    content=f"We have lots of information available for debugging an issue in our code below. Reproduce the info exactly as it is now, but exclude any useless information.\n\n{request.event_details.format_event()}"
                ),
            ],
            model="gpt-4o-mini-2024-07-18",
        )
        simplified_event_details = event_details_response.content

        state = self.context.state.get()

        response = agent.run(
            RootCauseAnalysisPrompts.format_default_msg(
                event=simplified_event_details,
                instruction=request.instruction,
                repo_names=[repo.full_name for repo in state.request.repos],
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

        formatter_response = agent.run(RootCauseAnalysisPrompts.root_cause_formatter_msg())

        with self.context.state.update() as cur:
            cur.usage += agent.usage - original_usage

        if not formatter_response:
            logger.warning("Root Cause Analysis formatter did not return a valid response")
            return None

        extracted_text = extract_text_inside_tags(formatter_response, "potential_root_causes")

        xml_response = RootCauseAnalysisOutputPromptXml.from_xml(
            f"<root><potential_root_causes>{escape_multi_xml(extracted_text, ['thoughts', 'title', 'description', 'code'])}</potential_root_causes></root>"
        )

        if not xml_response.potential_root_causes.causes:
            logger.warning("Root Cause Analysis formatter did not return causes")
            return None

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
