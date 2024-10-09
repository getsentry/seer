import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.dependency_injection import inject, injected


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

            You must output a title and description of the changes in the JSON, follow the format of:
            {{
                "title": "Title of the change",
                "description": "Description of the change"
            }}"""
        ).format(
            change_dump=change_dump,
            hint=hint,
        )


class ChangeDescriptionComponent(BaseComponent[ChangeDescriptionRequest, ChangeDescriptionOutput]):
    context: AutofixContext

    @observe(name="Change Describer")
    @ai_track(description="Change Describer")
    @inject
    def invoke(
        self, request: ChangeDescriptionRequest, llm_client: LlmClient = injected
    ) -> ChangeDescriptionOutput | None:
        output = llm_client.generate_structured(
            prompt=ChangeDescriptionPrompts.format_default_msg(
                change_dump=request.change_dump,
                hint=request.hint,
            ),
            model=OpenAiProvider.model("gpt-4o-mini"),
            response_format=ChangeDescriptionOutput,
        )
        data = output.parsed

        with self.context.state.update() as cur:
            cur.usage += output.metadata.usage

        if data is None:
            return None

        return data
