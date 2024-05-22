from sentry_sdk.ai.monitoring import ai_track
from lxml import etree

from seer.automation.agent.agent import GptAgent
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisOutput,
    RootCauseAnalysisOutputPromptXml,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.components.root_cause.prompts import RootCauseAnalysisPrompts
from seer.automation.autofix.tools import BaseTools
from seer.automation.autofix.utils import autofix_logger, escape_multi_xml
from seer.automation.component import BaseComponent


class RootCauseAnalysisComponent(BaseComponent[RootCauseAnalysisRequest, RootCauseAnalysisOutput]):
    context: AutofixContext

    @ai_track(description="Root Cause Analysis")
    def invoke(self, request: RootCauseAnalysisRequest) -> RootCauseAnalysisOutput | None:
        tools = BaseTools(self.context)

        agent = GptAgent(
            tools=tools.get_tools(),
            memory=[Message(role="system", content=RootCauseAnalysisPrompts.format_system_msg())],
        )

response = agent.run(
    RootCauseAnalysisPrompts.format_default_msg(
        err_msg=request.event_details.title,
        exceptions=request.event_details.exceptions,
        instruction=request.instruction,
    )
)

with self.context.state.update() as cur:
    cur.usage += agent.usage

if not response:
    autofix_logger.warning("Root Cause Analysis agent did not return a valid response")
    return None

sanitized_response = escape_multi_xml(response, ['thoughts', 'snippet', 'title', 'description'])
xml_content = f"&lt;root&gt;{sanitized_response}&lt;/root&gt;"

# Validate the XML content
try:
    etree.fromstring(xml_content)
except etree.XMLSyntaxError as e:
    raise ValueError(f"Malformed XML content: {e}")

xml_response = RootCauseAnalysisOutputPromptXml.from_xml(xml_content)

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
