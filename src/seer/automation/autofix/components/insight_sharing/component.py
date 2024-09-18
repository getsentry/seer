import re
import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import (
    InsightContextOutput,
    InsightSharingOutput,
    InsightSharingRequest,
)
from seer.automation.component import BaseComponent
from seer.automation.utils import extract_parsed_model
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

            Write the next under-25-words conclusion in the chain of thought based on the notes below, or if there is no good conclusion to add, return <NO_INSIGHT/>. The criteria for a good conclusion are that it should be a large, novel jump in insights, not similar to any item in the existing chain of thought, it should be a complete conclusion after analysis, it should not be a plan of what to analyze next, and it should be valuable for {task_description}. Every item in the chain of thought should read like a chain that clearly builds off of the previous step. If you can't find a conclusion that meets these criteria, return <NO_INSIGHT/>.

            {latest_thought}"""
        ).format(
            task_description=task_description,
            latest_thought=latest_thought,
            insights="\n".join(past_insights) if past_insights else "None",
        )

    @staticmethod
    def format_step_two(insight: str, latest_thought: str):
        return textwrap.dedent(
            """\
            Return the pieces of context from the issue details or the files in the codebase that are directly relevant to the text below:
            {insight}

            That means choose the most relevant codebase snippets, event logs, stacktraces, or other information, that show specifically what the text mentions. Don't include any repeated information; just include what's needed.

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
        self, request: InsightSharingRequest, gpt_client: GptClient = injected
    ) -> InsightSharingOutput | None:
        prompt_one = InsightSharingPrompts.format_step_one(
            task_description=request.task_description,
            latest_thought=request.latest_thought,
            past_insights=request.past_insights,
        )
        completion = gpt_client.openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[Message(role="user", content=prompt_one).to_message()],
            temperature=0.0,
        )
        with self.context.state.update() as cur:
            usage = Usage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            cur.usage += usage
        insight = completion.choices[0].message.content
        if insight == "<NO_INSIGHT/>":
            return None

        insight = re.sub(
            r"^\d+\.\s+", "", insight
        )  # since the model often starts the insight with a number, e.g. "3. Insight..."

        prompt_two = InsightSharingPrompts.format_step_two(
            insight=insight,
            latest_thought=request.latest_thought,
        )
        memory = []
        for i, message in enumerate(gpt_client.clean_tool_call_assistant_messages(request.memory)):
            if message.role != "system":
                memory.append(message.to_message())
        memory.append(Message(role="user", content=prompt_two).to_message())

        completion = gpt_client.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=memory,
            response_format=InsightContextOutput,
            temperature=0.0,
            max_tokens=2048,
        )
        with self.context.state.update() as cur:
            usage = Usage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            cur.usage += usage
        res = extract_parsed_model(completion)
        response = InsightSharingOutput(
            insight=insight,
            justification=res.explanation,
            error_message_context=res.error_message_context,
            codebase_context=res.codebase_context,
            stacktrace_context=res.stacktrace_context,
            breadcrumb_context=res.event_log_context,
        )
        return response
