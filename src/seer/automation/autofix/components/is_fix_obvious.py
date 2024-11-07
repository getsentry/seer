import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails
from seer.dependency_injection import inject, injected


class IsFixObviousRequest(BaseComponentRequest):
    event_details: EventDetails
    task_str: str
    fix_instruction: str | None
    memory: list[Message]


class IsFixObviousOutput(BaseComponentOutput):
    is_single_simple_change: bool


class IsFixObviousPrompts:
    @staticmethod
    def format_default_msg(
        event_details: EventDetails,
        task_str: str,
        fix_instruction: str | None,
    ):
        return (
            textwrap.dedent(
                """\
                Here is an issue in our codebase:

                {event_details}

                The root cause of the issue has been identified and context about the issue has been provided:
                {task_str}

                {fix_instruction}

                Is the code change simple and exists in only a single file?"""
            )
            .format(
                event_details=event_details.format_event(),
                task_str=task_str,
                fix_instruction=fix_instruction if fix_instruction else "",
            )
            .strip()
        )


class IsFixObviousComponent(BaseComponent[IsFixObviousRequest, IsFixObviousOutput]):
    context: AutofixContext

    @observe(name="Check if Obvious")
    @ai_track(description="Check if Obvious")
    @inject
    def invoke(
        self, request: IsFixObviousRequest, llm_client: LlmClient = injected
    ) -> IsFixObviousOutput | None:
        output = llm_client.generate_structured(
            messages=[
                Message(
                    role="user",
                    content=IsFixObviousPrompts.format_default_msg(
                        event_details=request.event_details,
                        task_str=request.task_str,
                        fix_instruction=request.fix_instruction,
                    ),
                ),
                *request.memory,
            ],
            model=OpenAiProvider.model("gpt-4o-mini"),
            response_format=IsFixObviousOutput,
        )
        data = output.parsed

        with self.context.state.update() as cur:
            cur.usage += output.metadata.usage

        if data is None:
            return IsFixObviousOutput(is_single_simple_change=False)
        return data
