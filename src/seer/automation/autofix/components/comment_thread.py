import textwrap

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.dependency_injection import inject, injected


class CommentThreadRequest(BaseComponentRequest):
    run_memory: list[Message] = []
    thread_memory: list[Message] = []
    selected_text: str | None = None


class CommentThreadOutput(BaseComponentOutput):
    comment_in_response: str
    asked_to_do_something: bool = False


class CommentThreadPrompts:
    @staticmethod
    def format_system_msg() -> str:
        return "You were an principle engineer responsible for debugging and fixing an issue in a codebase. You have memory of the previous conversation and analysis. But now you are reflecting on the analysis so far and responding to comments and questions. If you don't know something or got something wrong, say so. If you are sure of yourself, explain yourself.\n\nYou should also determine if the user is asking you to do something, i.e. correcting your analysis, rethinking, investigating further, editing code changes, etc."

    @staticmethod
    def format_default_msg(selected_text: str | None, thread_memory: list[Message]) -> str:
        selected_text = (
            f'the following statement from the analysis so far: "{selected_text}"'
            if selected_text
            else "the whole analysis"
        )
        msg = f"Ignore all previous instructions. Let us discuss {selected_text}. Here is the comment thread so far:\n\n"

        for message in thread_memory:
            prefix = "User said: " if message.role == "user" else "You said: "
            msg += f"{prefix}{message.content}\n"

        msg += "\n Now you respond briefly:"
        return textwrap.dedent(msg)


class CommentThreadComponent(BaseComponent[CommentThreadRequest, CommentThreadOutput]):
    context: AutofixContext

    @observe(name="Comment Thread")
    @sentry_sdk.trace
    @inject
    def invoke(
        self, request: CommentThreadRequest, llm_client: LlmClient = injected
    ) -> CommentThreadOutput | None:
        output = llm_client.generate_structured(
            prompt=CommentThreadPrompts.format_default_msg(
                selected_text=request.selected_text,
                thread_memory=request.thread_memory,
            ),
            messages=request.run_memory,
            system_prompt=CommentThreadPrompts.format_system_msg(),
            model=GeminiProvider.model("gemini-2.0-flash-001"),
            response_format=CommentThreadOutput,
        )
        data = output.parsed

        if data is None:
            return CommentThreadOutput(
                comment_in_response="Sorry, I'm not sure what to say.",
                action_requested=False,
            )
        return data
