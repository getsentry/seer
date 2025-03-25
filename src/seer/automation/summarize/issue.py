import textwrap

from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.autofixability import AutofixabilityModel
from seer.automation.models import EventDetails
from seer.automation.summarize.models import (
    GetFixabilityScoreRequest,
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

    @classmethod
    def from_db_state(cls, db_state: DbIssueSummary):
        item = cls.model_validate(db_state.summary)
        item.scores = SummarizeIssueScores(
            fixability_score=db_state.fixability_score,
            fixability_score_version=db_state.fixability_score_version,
            is_fixable=db_state.is_fixable,
            possible_cause_confidence=item.scores.possible_cause_confidence,
            possible_cause_novelty=item.scores.possible_cause_novelty,
        )
        return item

    def to_db_state(self, group_id: int):
        return DbIssueSummary(
            group_id=group_id,
            summary=self.model_dump(mode="json"),
            fixability_score=self.scores.fixability_score,
            fixability_score_version=self.scores.fixability_score_version,
            is_fixable=self.scores.is_fixable,
        )

    def to_summarize_issue_response(self, group_id: int):
        return SummarizeIssueResponse(
            group_id=group_id,
            headline=self.title,
            whats_wrong=self.whats_wrong,
            trace=self.session_related_issues,
            possible_cause=self.possible_cause,
            scores=self.scores,
        )


class IssueSummaryForLlmToGenerate(BaseModel):
    whats_wrong: str
    session_related_issues: str
    possible_cause: str
    possible_cause_novelty_score: float
    possible_cause_confidence_score: float
    title: str


@observe(name="Summarize Issue")
@inject
def summarize_issue(
    request: SummarizeIssueRequest, llm_client: LlmClient = injected
) -> IssueSummaryWithScores:
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
        Please write under 15 words per section, while cramming as much detail as possible, to make the report super easy to skim. Please bold important insights and unique terms by surrounding them with double asterisks (**like this**). The structure you should follow is:

        ###### What's wrong? [not optional]
        Summary of the stacktrace, breadcrumbs, and other context

        ###### Session related issues [optional]
        Insights from the application session issues, if relevant to this issue [you must return an empty string here if there are no relevant session related issues]

        ###### Possible cause [optional]
        Guess as to the cause, maybe show if there's clear smoking bullet. [return empty string if none]
        The guess should be somewhat novel compared to the `whats_wrong` section.
        If you're not sure about the guess, express this uncertainty subtly in your guess, e.g., by using words like "possible", "perhaps", "may be", etc.

        ###### Possible cause novelty score [optional]
        If you filled out the `possible_cause`, return a float score between 0 and 1 for how much new, insightful information is in the `possible_cause` section compared to the `whats_wrong` section.
        A `possible_cause` that is mostly a rephrasing of the `whats_wrong` section should score low.
        A `possible_cause` whose information is obviously implied by the `whats_wrong` section should score low.
        A `possible_cause` that contains new and insightful information that isn't mentioned in the `whats_wrong` section should score high.

        ###### Possible cause confidence score [optional]
        If you filled out the `possible_cause`, return a float score between 0 and 1 for how certain you are that this `possible_cause` is the true, upstream source of the issue. This score should be granular, e.g., 0.432.
        A guess that may contain erroneous information should score very low.
        A guess that contains a shred of speculation or vagueness should score low.
        A guess that only contains verifiably correct, specific, and unspeculative information should score high.
        Be hypercritical of your guess.
        If you didn't make a guess for the possible cause, return none here.

        ###### Title [not optional]
        Provide a concise title for the report that summarizes the `whats_wrong`, `session_related_issues`, and `possible_cause` sections.

        IMPORTANT: each section should be unique and not repeat the information in another section."""
    )

    completion = llm_client.generate_structured(
        model=GeminiProvider.model("gemini-2.0-flash-lite"),
        prompt=prompt,
        response_format=IssueSummaryForLlmToGenerate,
        temperature=0.0,
        max_tokens=512,
    )

    issue_summary = completion.parsed
    issue_summary_with_scores = IssueSummaryWithScores(
        **issue_summary.model_dump(),
        scores=SummarizeIssueScores(
            possible_cause_confidence=issue_summary.possible_cause_confidence_score,
            possible_cause_novelty=issue_summary.possible_cause_novelty_score,
        ),
    )

    return issue_summary_with_scores


def run_summarize_issue(request: SummarizeIssueRequest) -> SummarizeIssueResponse:
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

    summary = summarize_issue(request, **extra_kwargs)

    with Session() as session:
        db_state = summary.to_db_state(request.group_id)
        session.merge(db_state)
        session.commit()

    return summary.to_summarize_issue_response(request.group_id)


@observe(name="Get Fixability Score")
def run_fixability_score(
    request: GetFixabilityScoreRequest, autofixability_model: AutofixabilityModel
) -> SummarizeIssueResponse:
    langfuse_context.update_current_trace(session_id=f"group:{request.group_id}")
    with Session() as session:
        db_state = session.get(DbIssueSummary, request.group_id)
        if not db_state:
            raise ValueError(f"No issue summary found for group_id: {request.group_id}")
        issue_summary = IssueSummaryWithScores.from_db_state(db_state)

    fixability_score, is_fixable = evaluate_autofixability(issue_summary, autofixability_model)

    with Session() as session:
        issue_summary.scores.fixability_score = fixability_score
        issue_summary.scores.fixability_score_version = 2
        issue_summary.scores.is_fixable = is_fixable
        session.merge(issue_summary.to_db_state(request.group_id))
        session.commit()

    return issue_summary.to_summarize_issue_response(request.group_id)


@observe(name="Evaluate Autofixability")
def evaluate_autofixability(
    issue_summary: IssueSummaryWithScores, autofixability_model: AutofixabilityModel
) -> tuple[float, bool]:
    issue_summary_input = (
        f"Here's an issue:\n"
        f"Issue title: {issue_summary.title}\n"
        f"What's wrong: {issue_summary.whats_wrong}\n"
        f"Possible cause: {issue_summary.possible_cause}"
    )
    score = autofixability_model.score(issue_summary_input)
    is_fixable = score > 0.64727825  # 80th percentile
    return score, is_fixable
