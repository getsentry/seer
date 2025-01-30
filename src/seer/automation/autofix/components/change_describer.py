import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.dependency_injection import inject, injected


class ChangeDescriptionRequest(BaseComponentRequest):
    change_dump: str
    hint: str | None = None


class ChangeDescriptionOutput(BaseComponentOutput):
    title: str
    description: str


class ChangeDescriptionPrompts:
    @staticmethod
    def format_default_msg(
        change_dump: str,
        hint: str | None = None,
    ):
        return textwrap.dedent(
            """\
            Describe the following changes:

            {change_dump}

            {hint}
            You must output a title and description of the changes that are quickly readable for other engineers. Follow the format of:
            {{
                "title": "The most important specific change that is being made.",
                "description": "A brief bulleted list of the changes."
            }}"""
        ).format(
            change_dump=change_dump,
            hint=f"In the style of: {hint}\n" if hint else "",
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
            model=GeminiProvider.model("gemini-1.5-flash"),
            response_format=ChangeDescriptionOutput,
        )
        data = output.parsed

        with self.context.state.update() as cur:
            cur.usage += output.metadata.usage

        if data is None:
            return None

        return data
