import textwrap

from langfuse.decorators import observe
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput, InsightSharingRequest
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.dependency_injection import inject, injected


class InsightSharingPrompts:
    @staticmethod
    def format_default_msg(
        task_description: str,
        latest_thought: str,
    ):
        return textwrap.dedent(
            """\
            You're an engineer leading the process of {task_description}.
            To help your team, whenever you find an important insight needed for the process of {task_description}, you should document it. The only things we want to document are key conclusions in {task_description} that would belong in a final report, not random thoughts, tasks, or work-in-progress plans.

            You can look back on the conversation so far for context, but we're focused on the latest thought you had, which was:
            ---
            {latest_thought}
            ---

            First decide whether or not there is anything about this thought of yours that's important to share with your team and permanently document. If not, respond with no insight and no context items; should_share_insight is false.
            
            If you think there is something new and critical to know from this thought regarding {task_description}, document it it. When documenting, you should give a clear, concise, and concrete insight (1 short line). Then you should provide a clear, concise justification (1 short line) for your insight using concrete pieces of context, whether it's a snippet from the codebase, a line from the stacktrace, an event log, an error message, or something else. Finally, return the specific context you needed for your justification so your team can connect the dots easily. Only include the minimum necessary; leave out anything not critical for understanding your insight."""
        ).format(
            task_description=task_description,
            latest_thought=latest_thought
        )


class InsightSharingComponent(BaseComponent[InsightSharingRequest, InsightSharingOutput]):
    context: AutofixContext

    @observe(name="Sharing Insights")
    @ai_track(description="Sharing Insights")
    @inject
    def invoke(self, request: InsightSharingRequest, gpt_client: GptClient = injected) -> InsightSharingOutput | None:
        prompt = InsightSharingPrompts.format_default_msg(
            task_description=request.task_description,
            latest_thought=request.latest_thought,
        )

        memory = []
        for msg in request.memory:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                msg.role == "user"
            msg.role = "user" if msg.role == "tool" else msg.role
            msg.tool_calls = []
            msg.tool_call_id = None
            memory.append(msg.to_message())
        memory.append(Message(role="user", content=prompt).to_message())

        completion = gpt_client.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=memory,
            response_format=InsightSharingOutput,
            temperature=0.0,
            max_tokens=2048,
        )

        with self.context.state.update() as cur:
            usage = Usage(completion_tokens=completion.usage.completion_tokens, prompt_tokens=completion.usage.prompt_tokens, total_tokens=completion.usage.total_tokens)
            cur.usage += usage

        structured_message = completion.choices[0].message
        if structured_message.refusal:
            raise RuntimeError(structured_message.refusal)
        if not structured_message.parsed:
            raise RuntimeError("Failed to parse message")

        res = completion.choices[0].message.parsed
        return res
