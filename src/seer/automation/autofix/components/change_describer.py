import textwrap

from langsmith import traceable
from sentry_sdk.ai_analytics import ai_track

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest


class ChangeDescriptionRequest(BaseComponentRequest):
    change_dump: str
    hint: str


class ChangeDescriptionOutput(BaseComponentOutput):
    title: str
    description: str


class ChangeDescriptionPrompts:
    @staticmethod
    def format_default_msg(
        change_dump: str,
        hint: str,
    ):
        return textwrap.dedent(
            """\
            Describe the following changes:

            {change_dump}

            In the style of:
            {hint}

            You must output a title and description of the changes in the JSON."""
        ).format(
            change_dump=change_dump,
            hint=hint,
        )


class ChangeDescriptionComponent(BaseComponent[ChangeDescriptionRequest, ChangeDescriptionOutput]):
    context: AutofixContext

    @traceable(name="Change Describer", run_type="llm", tags=["change_describer:v1"])
    @ai_track(description="Change Describer")
    def invoke(self, request: ChangeDescriptionRequest) -> ChangeDescriptionOutput | None:
        prompt = ChangeDescriptionPrompts.format_default_msg(
            change_dump=request.change_dump,
            hint=request.hint,
        )

        data, message, usage = GptClient().json_completion(
            [Message(role="user", content=prompt)],
        )

        with self.context.state.update() as cur:
            cur.usage += usage

        if data is None or "title" not in data or "description" not in data:
            return None

        return ChangeDescriptionOutput(
            title=data.get("title", ""), description=data.get("description", "")
        )
