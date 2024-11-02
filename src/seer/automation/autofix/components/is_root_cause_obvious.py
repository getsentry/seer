import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails
from seer.dependency_injection import inject, injected


class IsRootCauseObviousRequest(BaseComponentRequest):
    event_details: EventDetails


class IsRootCauseObviousOutput(BaseComponentOutput):
    is_root_cause_clear: bool


class IsRootCauseObviousPrompts:
    @staticmethod
    def format_default_msg(
        event_details: EventDetails,
    ):
        return (
            textwrap.dedent(
                """\
            You are an exceptional principal engineer that is amazing at finding the root cause of any issue. We have an issue in our codebase described below. Is the root cause of the issue clear from the details below? Or does it require searching for more information around the codebase?

            {event_details}"""
            )
            .format(
                event_details=event_details.format_event(),
            )
            .strip()
        )


class IsRootCauseObviousComponent(
    BaseComponent[IsRootCauseObviousRequest, IsRootCauseObviousOutput]
):
    context: AutofixContext

    @observe(name="Check if Obvious")
    @ai_track(description="Check if Obvious")
    @inject
    def invoke(
        self, request: IsRootCauseObviousRequest, llm_client: LlmClient = injected
    ) -> IsRootCauseObviousOutput | None:
        output = llm_client.generate_structured(
            prompt=IsRootCauseObviousPrompts.format_default_msg(
                event_details=request.event_details,
            ),
            model=OpenAiProvider.model("gpt-4o-mini"),
            response_format=IsRootCauseObviousOutput,
        )
        data = output.parsed

        with self.context.state.update() as cur:
            cur.usage += output.metadata.usage

        if data is None:
            return IsRootCauseObviousOutput(is_root_cause_clear=False)
        return data
