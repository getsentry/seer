import re
import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import LlmClient, OpenAiProvider
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import (
    InsightContextOutput,
    InsightSharingOutput,
    InsightSharingRequest,
)
from seer.automation.component import BaseComponent
from seer.dependency_injection import inject, injected


class InsightSharingPrompts:
    @staticmethod
    def format_step_one(
        task_description: str,
        latest_thought: str,
        past_insights: list[str],
    ):
        past_insights = [f"{i + 1}. {insight}" for i, insight in enumerate(past_insights)]
        return textwrap.dedent(
            """\
            Given the chain of thought below for {task_description}:
            {insights}

            Write the next under-20-word conclusion in the chain of thought based on the notes below, or if there is no good conclusion to add, return <NO_INSIGHT/>. The criteria for a good conclusion are that it should be a large, novel jump in insights, not similar to any item in the existing chain of thought, it should be a complete conclusion after some meaty analysis, not a plan of what to analyze next, and it should be valuable for {task_description}. It should also be very concrete, to-the-point, and specific. Every item in the chain of thought should read like a chain that clearly builds off of the previous step. If you can't find a conclusion that meets ALL of these criteria, return <NO_INSIGHT/>.

            NOTES TO ANALYZE:
            {latest_thought}"""
        ).format(
            task_description=task_description,
            latest_thought=latest_thought,
            insights="\n".join(past_insights) if past_insights else "not started yet",
        )

    @staticmethod
    def format_step_two(insight: str, latest_thought: str):
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


class InsightSharingComponent(BaseComponent[InsightSharingRequest, InsightSharingOutput]):
    context: AutofixContext

    @observe(name="Sharing Insights")
    @ai_track(description="Sharing Insights")
    @inject
    def invoke(
        self, request: InsightSharingRequest, llm_client: LlmClient = injected
    ) -> InsightSharingOutput | None:
        try:
            prompt_one = InsightSharingPrompts.format_step_one(
                task_description=request.task_description,
                latest_thought=request.latest_thought,
                past_insights=request.past_insights,
            )
            completion = llm_client.generate_text(
                model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
                prompt=prompt_one,
                temperature=0.0,
            )
            with self.context.state.update() as cur:
                cur.usage += completion.metadata.usage
            insight = completion.message.content
            if not insight or insight == "<NO_INSIGHT/>":
                return None

            insight = re.sub(
                r"^\d+\.\s+", "", insight
            )  # since the model often starts the insight with a number, e.g. "3. Insight..."

            prompt_two = InsightSharingPrompts.format_step_two(
                insight=insight,
                latest_thought=request.latest_thought,
            )
            memory = []
            for message in llm_client.clean_tool_call_assistant_messages(request.memory):
                if message.role != "system":
                    memory.append(message)

            completion = llm_client.generate_structured(
                messages=memory,
                prompt=prompt_two,
                model=OpenAiProvider.model("gpt-4o-mini-2024-07-18"),
                response_format=InsightContextOutput,
                temperature=0.0,
                max_tokens=4096,
            )

            with self.context.state.update() as cur:
                cur.usage += completion.metadata.usage

            response = InsightSharingOutput(
                insight=insight,
                justification=completion.parsed.explanation,
                codebase_context=completion.parsed.codebase_context,
                stacktrace_context=completion.parsed.stacktrace_context,
                breadcrumb_context=completion.parsed.event_log_context,
                generated_at_memory_index=request.generated_at_memory_index,
            )
            return response
        except Exception:
            return None
