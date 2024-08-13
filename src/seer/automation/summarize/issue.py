import textwrap

from langfuse.decorators import observe
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from seer.automation.agent.client import GptClient
from seer.automation.models import EventDetails
from seer.automation.summarize.models import SummarizeIssueRequest, SummarizeIssueResponse
from seer.dependency_injection import inject, injected


class IssueSummary(BaseModel):
    cause_of_issue: str
    impact: str


@observe(name="Summarize Issue")
@inject
def summarize_issue(request: SummarizeIssueRequest, gpt_client: GptClient = injected):
    event_details = EventDetails.from_event(request.issue.events[0])

    prompt = textwrap.dedent(
        """\
        You are an exceptional developer that understands the issue and can summarize it in 1-2 sentences.
        {event_details}

        Analyze the issue, find the root cause, and summarize it in 1-2 sentences. In your answer, make sure to use backticks to highlight code snippets, output two results:

        # Cause of issue
        - 1 sentence, be extremely verbose with the exact snippets of code that are causing the issue.
        - Be extremely short and specific.
        - When talking about pieces of code, try to shorten it, so for example, instead of saying `foo.1.Def.bar` was undefined, say `Def` was undefined. Or saying if `foo.bar.baz.Class` is missing input field `bam.bar.Object` say `Class` is missing input field `Object`.
        - A developer that sees this should know exactly what to fix right away.

        # The impact on the system and users
        - 1 sentence, be extremely verbose with how this issue affects the system and end users.
        - Be extremely short and specific.

        Reason & explain the thought process step-by-step before giving the answers."""
    ).format(event_details=event_details.format_event())

    message_dicts: list[ChatCompletionMessageParam] = [
        {
            "content": prompt,
            "role": "user",
        },
    ]

    completion = gpt_client.openai_client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=message_dicts,
        temperature=0.0,
        max_tokens=2048,
    )

    message = completion.choices[0].message

    if message.refusal:
        raise RuntimeError(message.refusal)

    message_dicts.append(
        {
            "content": message.content,
            "role": "assistant",
        }
    )

    formatting_prompt = textwrap.dedent(
        """\
        Format your answer to the following schema."""
    )
    message_dicts.append(
        {
            "content": formatting_prompt,
            "role": "user",
        }
    )

    structured_completion = gpt_client.openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=message_dicts,
        temperature=0.0,
        max_tokens=2048,
        response_format=IssueSummary,
    )

    structured_message = structured_completion.choices[0].message

    if structured_message.refusal:
        raise RuntimeError(structured_message.refusal)

    if not structured_message.parsed:
        raise RuntimeError("Failed to parse message")

    return SummarizeIssueResponse(
        group_id=request.group_id,
        summary=structured_message.parsed.cause_of_issue,
        impact=structured_message.parsed.impact,
    )
