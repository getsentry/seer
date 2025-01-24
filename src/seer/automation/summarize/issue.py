import textwrap
from functools import cache

import numpy as np
import numpy.typing as npt
from langfuse.decorators import observe
from pydantic import BaseModel

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.models import EventDetails
from seer.automation.summarize import scoring
from seer.automation.summarize.models import (
    SummarizeIssueRequest,
    SummarizeIssueResponse,
    SummarizeIssueScores,
)
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


class IssueSummaryWithScores(IssueSummary):
    scores: SummarizeIssueScores


@cache
def _confidence_embeddings_cause() -> npt.NDArray[np.float64]:
    return scoring.embed_texts(["The cause is uncertain.", "The cause is certain."])


def score_issue_summary(issue_summary: IssueSummary) -> SummarizeIssueScores:
    # Embed everything we need once to minimize network requests.
    embedding_possible_cause, embedding_whats_wrong = scoring.embed_texts(
        [
            f"Cause: {issue_summary.possible_cause}",
            issue_summary.whats_wrong,
        ]
    )

    embeddings_confidence = _confidence_embeddings_cause()
    possible_cause_confidence = scoring.predict_proba(
        embedding_possible_cause, embeddings_confidence
    )[..., -1]
    # Extract the normalized score for the positive confidence
    # This score isn't a calibrated probability, but it's slightly useful with a threshold

    possible_cause_novelty = 1 - scoring.cosine_similarity(
        embedding_possible_cause, embedding_whats_wrong
    )

    return SummarizeIssueScores(
        possible_cause_confidence=possible_cause_confidence.item(),
        possible_cause_novelty=possible_cause_novelty,
    )


@observe(name="Summarize Issue")
@inject
def summarize_issue(
    request: SummarizeIssueRequest, llm_client: LlmClient = injected
) -> tuple[SummarizeIssueResponse, IssueSummaryWithScores]:
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

    issue_summary = completion.parsed
    issue_summary_with_scores = IssueSummaryWithScores(
        **issue_summary.model_dump(), scores=score_issue_summary(issue_summary)
    )

    return (
        SummarizeIssueResponse(
            group_id=request.group_id,
            headline=issue_summary_with_scores.title,
            whats_wrong=issue_summary_with_scores.whats_wrong,
            trace=issue_summary_with_scores.session_related_issues,
            possible_cause=issue_summary_with_scores.possible_cause,
            scores=issue_summary_with_scores.scores,
        ),
        issue_summary_with_scores,
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
