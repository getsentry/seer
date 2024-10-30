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
    is_fix_clear: bool


class IsFixObviousPrompts:
    @staticmethod
    def format_default_msg(
        event_details: EventDetails,
        task_str: str,
        fix_instruction: str | None,
    ):
        return textwrap.dedent(
            """\
            You are an exceptional principal engineer that is amazing at fixing any issue. We have an issue in our codebase and its root cause is described below, and you've already looked at some code files. Are the code changes needed to fix the issue clear from the details below? Or does it require searching for more information around the codebase?

            {event_details}

            The root cause of the issue has been identified and context about the issue has been provided:
            {task_str}

            {fix_instruction}"""
        ).format(
            event_details=event_details,
            task_str=task_str,
            fix_instruction=fix_instruction if fix_instruction else "",
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
            prompt=IsFixObviousPrompts.format_default_msg(
                event_details=request.event_details,
                task_str=request.task_str,
                fix_instruction=request.fix_instruction,
            ),
            messages=request.memory,
            model=OpenAiProvider.model("gpt-4o-mini"),
            response_format=IsFixObviousOutput,
        )
        data = output.parsed

        with self.context.state.update() as cur:
            cur.usage += output.metadata.usage

        if data is None:
            return IsFixObviousOutput(is_fix_clear=False)
        return data
