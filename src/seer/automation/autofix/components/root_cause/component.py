from langsmith import traceable

from seer.automation.agent.agent import GptAgent
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.root_cause.models import (
    MultipleRootCauseAnalysisOutputPromptXml,
    RootCauseAnalysisOutput,
    RootCauseAnalysisRequest,
)
from seer.automation.autofix.components.root_cause.prompts import RootCauseAnalysisPrompts
from seer.automation.autofix.tools import BaseTools
from seer.automation.autofix.utils import autofix_logger
from seer.automation.component import BaseComponent


class RootCauseAnalysisComponent(BaseComponent[RootCauseAnalysisRequest, RootCauseAnalysisOutput]):
    context: AutofixContext

    @traceable(name="Root Cause Analysis", run_type="llm", tags=["root-cause:v1"])
    def invoke(self, request: RootCauseAnalysisRequest) -> RootCauseAnalysisOutput | None:
        with self.context.state.update() as cur:
            tools = BaseTools(self.context)

            agent = GptAgent(
                tools=tools.get_tools(),
                memory=[
                    Message(role="system", content=RootCauseAnalysisPrompts.format_system_msg())
                ],
            )

            response = agent.run(
                RootCauseAnalysisPrompts.format_default_msg(
                    err_msg=request.event_details.title,
                    exceptions=request.event_details.exceptions,
                    instruction=request.instruction,
                )
            )

            cur.usage += agent.usage

            if not response:
                autofix_logger.warning("Root Cause Analysis agent did not return a valid response")
                return None

            xml_response = MultipleRootCauseAnalysisOutputPromptXml.from_xml(response)

            return RootCauseAnalysisOutput(
                causes=[cause.to_model() for cause in xml_response.causes]
            )
