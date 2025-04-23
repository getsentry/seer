import textwrap

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.dependency_injection import inject, injected


class InsightSharingPrompts:
    @staticmethod
    def format_step_one(
        *,
        latest_thought: str,
        past_insights: list[str],
        step_type: str,
    ):
        if step_type == "root_cause_analysis_processing":
            template = """\
            We have a new set of notes and thoughts:
            <NOTES>
            {latest_thought}
            </NOTES>

            We are in the process of writing a small paragraph describing the in-depth root cause of an issue in our codebase. Here it is so far:
            <PARAGRAPH_SO_FAR>
            {insights}
            </PARAGRAPH_SO_FAR>

            If there is something new and useful here to extend the root cause analysis, what is the NEXT sentence in the paragraph (write it so it flows nicely; max 15 words; write nothing but the sentence itself)? {no_insight_instruction}
            """
        elif step_type == "plan":
            template = """\
            We have a new set of notes and thoughts about the issue:
            <NOTES>
            {latest_thought}
            </NOTES>

            We are in the process of writing a small paragraph describing the code changes we need to make in our codebase. Here it is so far:
            <PARAGRAPH_SO_FAR>
            {insights}
            </PARAGRAPH_SO_FAR>

            If there is something new and useful here to extend the description on the code changes, what is the NEXT sentence in the paragraph (write it so it flows nicely; max 15 words; write nothing but the sentence itself)? {no_insight_instruction}
            """
        elif step_type == "solution_processing":
            template = """\
            We have a new set of notes and thoughts about the issue:
            <NOTES>
            {latest_thought}
            </NOTES>

            We are in the process of writing a small paragraph describing a solution to an issue in our codebase. Here it is so far:
            <PARAGRAPH_SO_FAR>
            {insights}
            </PARAGRAPH_SO_FAR>

            If there is something new and useful here to extend the solution proposal, what is the NEXT sentence in the paragraph (write it so it flows nicely; max 15 words; write nothing but the sentence itself)? {no_insight_instruction}
            """
        else:
            raise NotImplementedError(f"Insight sharing not implemented for step key: {step_type}")

        return textwrap.dedent(template).format(
            latest_thought=latest_thought,
            insights=" ".join(past_insights) if past_insights else "[paragraph is empty]",
            no_insight_instruction="If there is nothing SUPER IMPORTANT AND INSIGHTFUL to add, just return <NO_INSIGHT/>. If there is no new conclusion, but just a plan of what to search for, return <NO_INSIGHT/>. If this repeats something already in the paragraph, return <NO_INSIGHT/>.",
        )

    @staticmethod
    def format_step_two(*, insight: str, latest_thought: str):
        return textwrap.dedent(
            """\
            We had this thought:
            {latest_thought}

            And we concluded this:
            {insight}

            Now write one sentence of evidence backing the conclusion. And add any code, stacktraces, variable values, logs, or other evidence in Markdown code blocks using ```triple backticks``` when relevant to the conclusion."""
        ).format(
            insight=insight,
            latest_thought=latest_thought,
        )


@observe(name="Sharing Insights")
@sentry_sdk.trace
@inject
def create_insight_output(
    *,
    latest_thought: str,
    past_insights: list[str],
    step_type: str,
    memory: list[Message],
    generated_at_memory_index: int = -1,
    llm_client: LlmClient = injected,
) -> tuple[InsightSharingOutput | None, Usage]:
    usage = Usage()

    if not latest_thought or len(latest_thought.strip()) < 10:
        return None, usage

    prompt_one = InsightSharingPrompts.format_step_one(
        step_type=step_type,
        latest_thought=latest_thought,
        past_insights=past_insights,
    )

    completion = llm_client.generate_text(
        model=GeminiProvider.model("gemini-2.0-flash-001"),
        prompt=prompt_one,
        temperature=0.0,
    )

    insight = completion.message.content
    if (
        not insight
        or "<NO_INSIGHT/>" in insight
        or "</NO_INSIGHT/>" in insight
        or "</NO_INSIGHT>" in insight
    ):
        return None, usage

    prompt_two = InsightSharingPrompts.format_step_two(
        insight=insight,
        latest_thought=latest_thought,
    )

    memory = [msg for msg in memory if msg.role != "system"]

    completion = llm_client.generate_text(
        messages=memory,
        prompt=prompt_two,
        model=GeminiProvider.model("gemini-2.0-flash-001"),
        temperature=0.0,
        max_tokens=4096,
    )
    justification = completion.message.content

    usage += completion.metadata.usage

    response = InsightSharingOutput(
        insight=insight,
        justification=justification or "",
        generated_at_memory_index=generated_at_memory_index,
    )
    return response, usage
