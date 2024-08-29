import textwrap

from langfuse.decorators import observe
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from seer.automation.agent.client import GptClient
from seer.automation.models import EventDetails
from seer.automation.summarize.models import SummarizeIssueRequest, SummarizeIssueResponse
from seer.dependency_injection import inject, injected


class Step(BaseModel):
    reasoning: str
    justification: str


class IssueSummary(BaseModel):
    reason_step_by_step: list[Step]
    summary_of_the_issue_based_on_your_step_by_step_reasoning: str
    summary_of_the_functionality_affected: str
    five_to_ten_word_headline: str


@observe(name="Summarize Issue")
@inject
def summarize_issue(request: SummarizeIssueRequest, gpt_client: GptClient = injected):
    event_details = EventDetails.from_event(request.issue.events[0])
    connected_event_details = (
        [
            EventDetails.from_event(issue.events[0])
            for issue in request.connected_issues
            if issue.events
        ]
        if request.connected_issues
        else []
    )

    connected_issues_input = ""
    if connected_event_details:
        connected_issues = "\n----\n".join(
            [
                f"Connected Issue:\n{event.format_event()}"
                for _, event in enumerate(connected_event_details)
            ]
        )
        connected_issues_input = f"""
        Also, we know about some other issues that occurred in the same application trace, listed below. The issue above occurred somewhere alongside these:
        {connected_issues}
        """

    prompt = textwrap.dedent(
        f"""Our code is broken! Please summarize the issue below in a few short sentences so our engineers can immediately understand what's wrong and respond.

        The issue: {event_details.format_event()}

        {connected_issues_input}

        Your #1 goal is to help our engineers immediately understand the main issue and act!

        Regarding the issue summary, state clearly and concisely what's going wrong and why. Do not just restate the error message, as that wastes our time! Look deeper into the details provided to paint the full picture and find the key insight of what's going wrong.

        Our engineers need to get into the nitty gritty mechanical details of what's going wrong. The insight may be in the stacktrace, error message, event logs, or connected issues. Highlight the most important details across all the context you have that are relevant to the main issue!

        Regarding affected functionality, your goal is to help our engineers immediately understand what SPECIFIC application or service functionality is related to this code issue. Do NOT try to conclude root causes or suggest solutions. Don't even talk about the mechanical details, but rather speak more to the overall task that is affected. Don't comment on the severity of the issue or user impact."""
    )

    message_dicts: list[ChatCompletionMessageParam] = [
        {
            "content": prompt,
            "role": "user",
        },
    ]

    completion = gpt_client.openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=message_dicts,
        response_format=IssueSummary,
        temperature=0.0,
        max_tokens=2048,
    )
    structured_message = completion.choices[0].message
    if structured_message.refusal:
        raise RuntimeError(structured_message.refusal)
    if not structured_message.parsed:
        raise RuntimeError("Failed to parse message")

    res = completion.choices[0].message.parsed
    summary = res.summary_of_the_issue_based_on_your_step_by_step_reasoning
    impact = res.summary_of_the_functionality_affected
    headline = res.five_to_ten_word_headline

    return SummarizeIssueResponse(
        group_id=request.group_id,
        headline=headline,
        summary=summary,
        impact=impact,
    )


def run_summarize_issue(request: SummarizeIssueRequest):
    langfuse_tags = []
    if request.organization_slug:
        langfuse_tags.append(f"org:{request.organization_slug}")
    if request.project_id:
        langfuse_tags.append(f"project:{request.project_id}")
    if request.group_id:
        langfuse_tags.append(f"group:{request.group_id}")

    extra_kwargs = {
        "langfuse_tags": langfuse_tags,
        "langfuse_session_id": f"group:{request.group_id}",
        "langfuse_user_id": (
            f"org:{request.organization_slug}" if request.organization_slug else None
        ),
    }

    return summarize_issue(request, **extra_kwargs)
