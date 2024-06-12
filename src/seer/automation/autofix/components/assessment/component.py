from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

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

    @observe(name="Problem Discovery")
    @ai_track(description="Problem Discovery")
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
                            instruction=request.instruction,
                        ),
                    ),
                ],
            )

            cur.usage += usage

            if data is None:
                autofix_logger.warning("Problem discovery agent did not return a valid response")
                return None

            return ProblemDiscoveryOutput.model_validate(data)
