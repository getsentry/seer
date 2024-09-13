import textwrap
import re
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput, InsightSharingRequest
from seer.automation.component import BaseComponent
from seer.dependency_injection import inject, injected


class InsightSharingPrompts:
    @staticmethod
    def format_default_msg(
        task_description: str,
        latest_thought: str,
        past_insights: list[str],
    ):
        past_insights = [f"{i + 1}. {insight}" for i, insight in enumerate(past_insights)]
        return textwrap.dedent(
            """\
            Consider the last thing you said in the conversation. Is there any key takeaway insight in there that we should use to continue the WIP line of reasoning below?
            {insights}
              
            Make your answer 1 line that will be added onto the current line of reasoning. Separately, give a justification that should use context from the codebase and the issue details to quickly explain your insight. Also answer with snippets of only the most relevant context that helps explain.
            
            Finally, check:
            - if your insight turns out to be unimportant for {task_description}.
            - if your insight just repeats an already covered idea.
            - if your insight is an incomplete idea that isn't ready to be added to the list."""
        ).format(
            task_description=task_description,
            latest_thought=latest_thought,
            insights="\n".join(past_insights) if past_insights else "None",
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
            past_insights=request.past_insights,
        )

        memory = [
            message.to_message()
            for message in gpt_client.clean_tool_call_assistant_messages(request.memory) if message.role != "system"
        ]
        memory.append(Message(role="user", content=prompt).to_message())

        completion = gpt_client.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",#"gpt-4o-2024-08-06",
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
        res.insight = re.sub(r'^\d+\.\s+', '', res.insight) # since the model often starts the insight with a number, e.g. "3. Insight..."
        return res
