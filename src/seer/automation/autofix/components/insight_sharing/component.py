import re
import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.components.insight_sharing.models import (
    InsightContextOutput,
    InsightSharingOutput,
)
from seer.dependency_injection import inject, injected


class InsightSharingPrompts:
    @staticmethod
    @inject
    def format_step_one(
        *,
        task_description: str,
        latest_thought: str,
        past_insights: list[str],
        step_type: str | None = None,
    ):
        no_insight_instruction = (
            "If there is no clear conclusion that adds a ton of new insight and value, return <NO_INSIGHT/>. If it is similar to a previous conclusion, return <NO_INSIGHT/>."
            if past_insights
            else ""
        )

        past_insights = [f"{i + 1}. {insight}" for i, insight in enumerate(past_insights)]

        if step_type == "root_cause_analysis":
            template = """\
            Given the chain of thought below for {task_description}:
            {insights}

            Write the next under-20-word conclusion in the chain of thought based on the notes below. {no_insight_instruction} The criteria for a good conclusion are that it should be a large, novel jump in insights, not similar to any item in the existing chain of thought, it should be a complete conclusion after some meaty analysis, not a plan of what to analyze next, and it should be valuable for {task_description}. It should also be very concrete, to-the-point, and specific. Every item in the chain of thought should read like a chain that clearly builds off of the previous step. If you can't find a conclusion that meets ALL of these criteria, return <NO_INSIGHT/>.

            If you do write something, write it so that someone else could immediately understand your point in detail.

            NOTES TO ANALYZE:
            {latest_thought}"""
        elif step_type == "plan":
            template = """\
            Given the chain of thought below for {task_description}:
            {insights}

            Write the next under-20-word conclusion in the chain of thought based on the notes below, focusing specifically on building up to a solution. The conclusion should be a concrete plan or insight about how to fix the issue, not just analysis. {no_insight_instruction} Every item should build off the previous one towards the final solution.

            If you do write something, write it so that someone else could immediately understand your point in detail.

            NOTES TO ANALYZE:
            {latest_thought}"""
        else:
            template = """\
            Given the chain of thought below for {task_description}:
            {insights}

            Write the next under-20-word conclusion in the chain of thought based on the notes below. {no_insight_instruction} The criteria for a good conclusion are that it should be a large, novel jump in insights, not similar to any item in the existing chain of thought, it should be a complete conclusion after some meaty analysis, not a plan of what to analyze next, and it should be valuable for {task_description}. It should also be very concrete, to-the-point, and specific. Every item in the chain of thought should read like a chain that clearly builds off of the previous step.

            If you do write something, write it so that someone else could immediately understand your point in detail.

            NOTES TO ANALYZE:
            {latest_thought}"""

        return textwrap.dedent(template).format(
            task_description=task_description,
            latest_thought=latest_thought,
            insights="\n".join(past_insights) if past_insights else "not started yet",
            no_insight_instruction=no_insight_instruction,
        )

    @staticmethod
    @inject
    def format_step_two(*, insight: str, latest_thought: str):
        return textwrap.dedent(
            """\
            Return the pieces of context from the issue details or the files in the codebase that are directly relevant to the text below:
            {insight}

            That means choose the most relevant codebase snippets (codebase_context), event logs (breadcrumb_context), or stacktrace/variable data (stacktrace_context), that show specifically what the text mentions. Don't include any repeated information; just include what's needed.

            Also provide a one-line explanation of how the pieces of context directly explain the text.

            To know what's needed, reference these notes:
            {latest_thought}"""
        ).format(
            insight=insight,
            latest_thought=latest_thought,
        )


@observe(name="Sharing Insights")
@ai_track(description="Sharing Insights")
@inject
def create_insight_output(
    *,
    latest_thought: str,
    task_description: str,
    past_insights: list[str],
    step_type: str | None = None,
    memory: list[Message],
    generated_at_memory_index: int = -1,
    llm_client: LlmClient = injected,
) -> tuple[InsightSharingOutput | None, Usage]:
    usage = Usage()

    prompt_one = InsightSharingPrompts.format_step_one(
        step_type=step_type,
        task_description=task_description,
        latest_thought=latest_thought,
        past_insights=past_insights,
    )

    completion = llm_client.generate_text(
        model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
        prompt=prompt_one,
        temperature=0.0,
    )

    insight = completion.message.content
    if not insight or insight == "<NO_INSIGHT/>":
        return None, usage

    insight = re.sub(
        r"^\d+\.\s+", "", insight
    )  # since the model often starts the insight with a number, e.g. "3. Insight..."

    prompt_two = InsightSharingPrompts.format_step_two(
        insight=insight,
        latest_thought=latest_thought,
    )

    memory = [
        message
        for message in llm_client.clean_tool_call_assistant_messages(memory)
        if message.role != "system"
    ]

    completion = llm_client.generate_structured(
        messages=memory,
        prompt=prompt_two,
        model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
        response_format=InsightContextOutput,
        temperature=0.0,
        max_tokens=4096,
    )

    usage += completion.metadata.usage

    response = InsightSharingOutput(
        insight=insight,
        justification=completion.parsed.explanation,
        codebase_context=completion.parsed.codebase_context,
        stacktrace_context=completion.parsed.stacktrace_context,
        breadcrumb_context=completion.parsed.event_log_context,
        generated_at_memory_index=generated_at_memory_index,
    )
    return response, usage
