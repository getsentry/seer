import textwrap

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


class ChangeDescriptionRequest(BaseComponentRequest):
    change_dump: str
    hint: str | None = None
    previous_commits: list[str] | None = None


class ChangeDescriptionOutput(BaseComponentOutput):
    title: str
    description: str
    branch_name: str


class ChangeDescriptionPrompts:
    @staticmethod
    def format_default_msg(
        change_dump: str,
        hint: str | None = None,
        previous_commits: list[str] | None = None,
    ):
        formatting_instructions = "The title should be all lowercase except for symbol/variable names, prefixed with a 'fix:' prefix, and describe the change in a way that is easy to understand."
        if previous_commits:
            joined_commits = ", ".join(f'"{commit}"' for commit in previous_commits)
            formatting_instructions = f"Describe the change in a way that is easy to understand, closely following the formatting of previous commit titles in the repo, such as: {joined_commits}"

        return textwrap.dedent(
            """\
            Describe the following changes:

            {change_dump}

            {hint}
            You must output a title and description of the changes that are quickly readable for other engineers. Follow the format of:

            - Title: The most important specific change that is being made. {formatting_instructions}
            - Description: A brief bulleted list of the changes.
            - Branch Name: A short name for the branch that will be created to make the changes"""
        ).format(
            change_dump=change_dump,
            hint=f"In the style of: {hint}\n" if hint else "",
            formatting_instructions=formatting_instructions,
        )


class ChangeDescriptionComponent(BaseComponent[ChangeDescriptionRequest, ChangeDescriptionOutput]):
    context: AutofixContext

    @observe(name="Change Describer")
    @sentry_sdk.trace
    @inject
    def invoke(
        self,
        request: ChangeDescriptionRequest,
        llm_client: LlmClient = injected,
        config: AppConfig = injected,
    ) -> ChangeDescriptionOutput | None:
        de_config = {
            "models": [
                GeminiProvider.model("gemini-2.0-flash-001", region="europe-west1"),
                GeminiProvider.model("gemini-2.0-flash-001", region="europe-west4"),
            ],
        }
        us_config = {
            "models": [
                GeminiProvider.model("gemini-2.0-flash-001", region="us-central1"),
                GeminiProvider.model("gemini-2.0-flash-001", region="us-east1"),
            ],
        }

        output = llm_client.generate_structured(
            prompt=ChangeDescriptionPrompts.format_default_msg(
                change_dump=request.change_dump,
                hint=request.hint,
                previous_commits=request.previous_commits,
            ),
            response_format=ChangeDescriptionOutput,
            **(de_config if config.SENTRY_REGION == "de" else us_config),
        )
        data = output.parsed

        data.branch_name = f"seer/{data.branch_name}"

        with self.context.state.update() as cur:
            cur.usage += output.metadata.usage

        if data is None:
            return None

        return data
