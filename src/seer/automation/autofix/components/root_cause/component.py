import logging
import textwrap

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

        state = self.context.state.get()

        try:
            response = agent.run(
                RootCauseAnalysisPrompts.format_default_msg(
                    event=request.event_details.format_event(),
                    summary=request.summary,
                    instruction=request.instruction,
                    repo_names=[repo.full_name for repo in state.request.repos],
                ),
                context=self.context,
            )

            if not response:
                logger.warning("Root Cause Analysis agent did not return a valid response")
                return None

            if "<NO_ROOT_CAUSES>" in response:
                return None

            # Ask for reproduction
            agent.run(
                textwrap.dedent(
                    """\
                    Given all the above potential root causes you just gave, please provide a 1-2 sentence concise instruction on how to reproduce the issue for each root cause.
                    - Assume the user is an experienced developer well-versed in the codebase, simply give the reproduction steps.
                    - You must use the local variables provided to you in the stacktrace to give your reproduction steps.
                    - Try to be open ended to allow for the most flexibility in reproducing the issue. Avoid being too confident.
                    - This step is optional, if you're not sure about the reproduction steps for a root cause, just skip it."""
                )
            )

            def clean_tool_call_assistant_messages(messages: list[Message]):
                new_messages = []
                for message in messages:
                    if message.role == "assistant" and message.tool_calls:
                        new_messages.append(
                            Message(role="assistant", content=message.content, tool_calls=[])
                        )
                    elif message.role == "tool":
                        new_messages.append(
                            Message(role="user", content=message.content, tool_calls=[])
                        )
                    else:
                        new_messages.append(message)
                return new_messages

            response = GptClient().openai_client.beta.chat.completions.parse(
                messages=[
                    message.to_message()
                    for message in clean_tool_call_assistant_messages(agent.memory)
                ]
                + [
                    Message(
                        role="user",
                        content=RootCauseAnalysisPrompts.root_cause_formatter_msg(),
                    ).to_message(),  # type: ignore
                ],
                model="gpt-4o-2024-08-06",
                response_format=MultipleRootCauseAnalysisOutputPrompt,
            )

            parsed = extract_parsed_model(response)

            # Assign the ids to be the numerical indices of the causes and relevant code context
            causes = []
            for i, cause in enumerate(parsed.causes):
                cause_model = cause.to_model()
                cause_model.id = i

                if cause_model.code_context:
                    for j, snippet in enumerate(cause_model.code_context):
                        snippet.id = j

                causes.append(cause_model)

            return RootCauseAnalysisOutput(causes=causes)
        finally:
            with self.context.state.update() as cur:
                cur.usage += agent.usage
