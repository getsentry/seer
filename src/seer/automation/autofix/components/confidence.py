import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.dependency_injection import inject, injected


class ConfidenceRequest(BaseComponentRequest):
    run_memory: list[Message] = []
    step_goal_description: str
    next_step_goal_description: str


class ConfidenceOutput(BaseComponentOutput):
    comment: str | None = None
    output_confidence_score: float
    proceed_confidence_score: float


class ConfidencePrompts:
    @staticmethod
    def format_system_msg() -> str:
        return "You were an principle engineer responsible for debugging and fixing an issue in a codebase. You have memory of the previous conversation and analysis. But now you are reflecting on the analysis so far. Your goal is to verbalize any uncertainties that affected your answer, your confidence in your final answer, and your confidence in proceeding to the next step. You will decide whether to leave a brief comment on the document for your team to respond to, but only if significant concerns remain. You will also score your confidence."

    @staticmethod
    def format_default_msg(step_goal_description: str, next_step_goal_description: str) -> str:
        return textwrap.dedent(
            """\
            Think through the uncertainties and open questions, if any, that appeared during your analysis. Is there a missing piece of the puzzle? Anywhere you had to make an assumption or speculate? Any opportunities for a better answer? Anywhere you need more context or an opinion from the team? Be hypercritical. If there are uncertainties or open questions your team should be aware of when reading your final answer, leave a brief (under 50 words) and specific comment/question on the document. If there is nothing worth surfacing, return None/null for the comment.

            Then score your confidence in the correctness of your final {step_goal_description} with an float between 0 and 1. The more uncertainties there are, the lower your confidence should be.
            Then based on your findings so far, score your confidence in successfully completing the next step, {next_step_goal_description}, with an float between 0 and 1. The more uncertain you are about your correctness, or if it seems hard to do the next step based on what you know, the lower your confidence should be.
            Your scores should be granular, e.g., 0.432.
            """
        ).format(
            step_goal_description=step_goal_description,
            next_step_goal_description=next_step_goal_description,
        )


class ConfidenceComponent(BaseComponent[ConfidenceRequest, ConfidenceOutput]):
    context: AutofixContext

    @observe(name="Confidence")
    @ai_track(description="Confidence")
    @inject
    def invoke(
        self, request: ConfidenceRequest, llm_client: LlmClient = injected
    ) -> ConfidenceOutput | None:
        output = llm_client.generate_structured(
            prompt=ConfidencePrompts.format_default_msg(
                step_goal_description=request.step_goal_description,
                next_step_goal_description=request.next_step_goal_description,
            ),
            messages=request.run_memory,
            system_prompt=ConfidencePrompts.format_system_msg(),
            model=GeminiProvider.model("gemini-2.0-flash-001"),
            response_format=ConfidenceOutput,
        )
        data = output.parsed

        if data is None:
            return ConfidenceOutput(
                output_confidence_score=0.5,
                proceed_confidence_score=0.5,
            )
        return data
