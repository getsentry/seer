from langsmith import traceable

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.assessment.models import (
    ProblemDiscoveryOutput,
    ProblemDiscoveryRequest,
)
from seer.automation.autofix.components.assessment.prompts import ProblemDiscoveryPrompts
from seer.automation.autofix.utils import autofix_logger
from seer.automation.component import BaseComponent


class ProblemDiscoveryComponent(BaseComponent[ProblemDiscoveryRequest, ProblemDiscoveryOutput]):
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @traceable(name="Problem Discovery", run_type="llm", tags=["problem_discovery:v1.1"])
    def invoke(self, request: ProblemDiscoveryRequest) -> ProblemDiscoveryOutput | None:
        with self.context.state.update() as cur:
            gpt_client = GptClient()

            exceptions = request.event_details.exceptions

            data, message, usage = gpt_client.json_completion(
                [
                    Message(
                        role="system",
                        content=ProblemDiscoveryPrompts.format_system_msg(),
                    ),
                    Message(
                        role="user",
                        content=ProblemDiscoveryPrompts.format_default_msg(
                            event_title=request.event_details.title,
                            exceptions=exceptions,
                            additional_context=request.additional_context,
                        ),
                    ),
                ],
            )

            print("cur", cur)

            cur.usage += usage

            if data is None:
                autofix_logger.warning(f"Problem discovery agent did not return a valid response")
                return None

            return ProblemDiscoveryOutput.model_validate(data)
