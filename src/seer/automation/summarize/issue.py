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
    summary_of_issue: str
    affects_what_functionality: str
    known_customer_impact: str
    customer_impact_is_known: bool


@observe(name="Summarize Issue")
@inject
def summarize_issue(request: SummarizeIssueRequest, gpt_client: GptClient = injected):
    event_details = EventDetails.from_event(request.issue.events[0])

    prompt = textwrap.dedent(
        '''Our code is broken! Please summarize the issue below in 1 sentence so our engineers can immediately understand what's wrong and respond.
        {event_details}

        Your #1 goal is to help our engineers immediately understand the issue and act!
        Regarding the issue summary, state clearly and concisely what is going wrong in the code and why.
        Regarding affected functionality, your goal is to help our engineers immediately understand what SPECIFIC application or service functionality is related to this code issue. Do NOT try to conclude root causes or suggest solutions. Use a complete sentence.
        Regarding known impact, if you don't know for sure the impact, just say "Not enough information to assess user impact."'''
    ).format(event_details=event_details.format_event())

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
    summary = res.summary_of_issue
    impact = res.affects_what_functionality
    if (
        "Not enough information to assess user impact" not in res.known_customer_impact
        or res.customer_impact_is_known
    ):
        impact += f" {res.known_customer_impact}"

    return SummarizeIssueResponse(
        group_id=request.group_id,
        summary=summary,
        impact=impact,
    )
