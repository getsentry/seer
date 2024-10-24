import textwrap

from langfuse.decorators import observe
from pydantic import BaseModel

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.models import EventDetails
from seer.automation.summarize.models import SummarizeIssueRequest, SummarizeIssueResponse
from seer.db import DbIssueSummary, Session
from seer.dependency_injection import inject, injected


class Step(BaseModel):
    reasoning: str
    justification: str


class IssueSummary(BaseModel):
    reason_step_by_step: list[Step]
    bulleted_summary_of_the_issue_based_on_your_step_by_step_reasoning: str
    five_to_ten_word_headline: str


@observe(name="Summarize Issue")
@inject
def summarize_issue(
    request: SummarizeIssueRequest, llm_client: LlmClient = injected
) -> tuple[SummarizeIssueResponse, IssueSummary]:
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
        f"""Our code is broken! Please summarize the issue below in a few short bullet points so our engineers can immediately understand what's wrong and respond.

        The issue: {event_details.format_event()}

        {connected_issues_input}

        Your #1 goal is to help our engineers immediately understand the main issue and act!

        Regarding the issue details summary, state clearly and concisely what's going wrong and why. Do not just restate the error message, as that wastes our time! Look deeper into the details provided to paint the full picture and find the key insight of what's going wrong.

        Our engineers need to get into the nitty gritty mechanical details of what's going wrong. The insight may be in the stacktrace, error message, event logs, or connected issues. Highlight the most important details across all the context you have that are relevant to the main issue!

        Format all responses as multiple Markdown bullet points with shorthand language."""
    )

    completion = llm_client.generate_structured(
        model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
        prompt=prompt,
        response_format=IssueSummary,
        temperature=0.0,
        max_tokens=2048,
    )

    summary = completion.parsed.bulleted_summary_of_the_issue_based_on_your_step_by_step_reasoning
    impact = ""  # not generating impact for now. We'll revisit when we have a credible source to decide impact.
    headline = completion.parsed.five_to_ten_word_headline
    if headline.endswith(".") or headline.endswith("!"):
        headline = headline[:-1]
    headline = headline + "."

    return (
        SummarizeIssueResponse(
            group_id=request.group_id,
            headline=headline,
            summary=summary,
            impact=impact,
        ),
        completion.parsed,
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

    summary, raw_summary = summarize_issue(request, **extra_kwargs)

    with Session() as session:
        db_state = DbIssueSummary(
            group_id=request.group_id, summary=raw_summary.model_dump(mode="json")
        )
        session.merge(db_state)
        session.commit()

    return summary
