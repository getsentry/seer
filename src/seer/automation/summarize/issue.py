import textwrap

from langfuse.decorators import observe
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from seer.automation.agent.client import GptClient
from seer.automation.models import EventDetails
from seer.automation.summarize.models import SummarizeIssueRequest, SummarizeIssueResponse
from seer.dependency_injection import inject, injected


class IssueSummary(BaseModel):
    reason_step_by_step: str
    summary_of_issue_at_code_level: str
    summary_of_functionality_touched: str
    factual_issue_description_under_10_words: str


class IssueSummaryWithTrace(BaseModel):
    one_sentence_summary_of_main_issue_at_code_level: str
    one_sentence_summary_of_connected_issues_at_code_level: str
    insights_from_trace_at_code_level: str
    final_summary: str
    summary_of_functionality_touched: str
    factual_issue_description_under_10_words: str


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
    trace_summary_prompt = ""
    final_summary_trace_details = ""
    if connected_event_details:
        connected_issues = "\n----\n".join(
            [
                f"Connected Issue:\n{event.format_event()}"
                for _, event in enumerate(connected_event_details)
            ]
        )
        connected_issues_input = f"""
        Also, we know about some other issues that occurred in the same application trace, listed below. This issue occurred somewhere alongside these:
        {connected_issues}
        """

        trace_summary_prompt = """- Summarize the main issue at a code level standalone.
        - Summarize the connected issues at a code level standalone.
        - Describe insights from the trace; i.e. is this issue caused by another issue, or is it causing another issue, or are there other issues following the same pattern with some variations? Specifically mention other issues in the trace if there is anything to conclude. Be specific with code details and how these issues interact if at all."""

        final_summary_trace_details = "Make connections to details from the trace if they will be useful to our engineers in understanding this issue."

    prompt = textwrap.dedent(
        f"""Our code is broken! Please summarize the issue below in a few short sentences so our engineers can immediately understand what's wrong and respond.

        The issue: {event_details.format_event()}

        {connected_issues_input}

        Your #1 goal is to help our engineers immediately understand the main issue and act! Follow the below plan, giving detailed yet concise answers:

        {trace_summary_prompt}
        - Write a 1-2 line final summary of the specific code details of the issue. State clearly and concisely what's going wrong and why. Do not just restate the error message, as that wastes our time! Look deeper into the details provided to paint the full picture and find the key insight of what's going wrong. At the code level, our engineers need to get into the nitty gritty mechanical details of what's going wrong. {final_summary_trace_details}
        - Summarize what SPECIFIC functionality is touched by the issue. Do NOT try to conclude root causes or suggest solutions. Don't even talk about the mechanical details, but rather speak more to the overall task that is failing. Get straight to the point!
        - Write a professional headline-like summary of the overall issue."""
    )

    message_dicts: list[ChatCompletionMessageParam] = [
        {
            "content": "You speak only in news headlines (you can use multiple headlines for multiple sentences). Only use words that add value to our engineers who are debugging the issue under a time crunch. The more specific and granular you are, the faster our engineers can act. Handwavy broad talk about the service or application is a waste of time.",
            "role": "system",
        },
        {
            "content": prompt,
            "role": "user",
        },
    ]

    completion = gpt_client.openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=message_dicts,
        response_format=IssueSummaryWithTrace if connected_event_details else IssueSummary,
        temperature=0.0,
        max_tokens=2048,
    )
    structured_message = completion.choices[0].message
    if structured_message.refusal:
        raise RuntimeError(structured_message.refusal)
    if not structured_message.parsed:
        raise RuntimeError("Failed to parse message")

    res = completion.choices[0].message.parsed
    summary = res.final_summary if connected_event_details else res.summary_of_issue_at_code_level
    impact = res.summary_of_functionality_touched
    headline = res.factual_issue_description_under_10_words

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
