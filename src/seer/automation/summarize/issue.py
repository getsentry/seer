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
    title: str
    whats_wrong: str
    session_related_issues: str
    possible_cause: str


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
                f"Issue from the same session:\n{event.format_event()}"
                for _, event in enumerate(connected_event_details)
            ]
        )
        connected_issues_input = f"""
        Also, we know about some other issues that occurred in the same application session, listed below. The issue above occurred somewhere alongside these:
        {connected_issues}
        """
    else:
        connected_issues_input = "Issues from the same session: none"

    prompt = textwrap.dedent(
        f"""Our code is broken! Please summarize the issue below in a few short bullet points so our engineers can immediately understand what's wrong.

        The issue: {event_details.format_event()}

        {connected_issues_input}

        Your #1 goal is to help our engineers immediately understand what's going on in the main issue.

        Write a concise report that sets the scene for the issue. Know that our engineers can already see all the same details on the issue that were provided to you, so DO NOT repeat them. Instead extract the holistic insights that would take time to figure out otherwise.
        Please write under 20 words per section, while cramming as much detail as possible, to make the report super easy to skim. Please bold important insights and unique terms by surrounding them with double asterisks (**like this**). The structure you should follow is:

        ###### What's wrong? [not optional]
        summary of the stacktrace, breadcrumbs, and other context

        ###### Session related issues [optional]
        insights from the application session issues, if relevant to this issue [return empty string if none]

        ###### Possible cause [optional]
        guess as to the cause, maybe show if there's clear smoking bullet [return empty string if none]

        Also provide a concise title for the report that summarizes everything you write above.

        IMPORTANT: each section should be unique and not repeat the information in another section."""
    )

    completion = llm_client.generate_structured(
        model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
        prompt=prompt,
        response_format=IssueSummary,
        temperature=0.0,
        max_tokens=2048,
        timeout=7.0,
    )

    return (
        SummarizeIssueResponse(
            group_id=request.group_id,
            headline=completion.parsed.title,
            whats_wrong=completion.parsed.whats_wrong,
            trace=completion.parsed.session_related_issues,
            possible_cause=completion.parsed.possible_cause,
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
