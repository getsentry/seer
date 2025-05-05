import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, GeminiProvider, LlmClient
from seer.automation.autofix.tools.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodePrReviewOutput, CodePrReviewRequest
from seer.automation.codegen.prompts import CodingCodeReviewPrompts, CodingUnitTestPrompts
from seer.automation.component import BaseComponent
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class PrReviewCodingComponent(BaseComponent[CodePrReviewRequest, CodePrReviewOutput]):
    context: CodegenContext

    @observe(name="Review PR")
    @ai_track(description="Review PR")
    @inject
    def invoke(
        self, request: CodePrReviewRequest, llm_client: LlmClient = injected
    ) -> CodePrReviewOutput | None:
        with BaseTools(self.context, repo_client_type=RepoClientType.CODECOV_PR_REVIEW) as tools:
            agent = LlmAgent(
                tools=tools.get_tools(),
                config=AgentConfig(interactive=False),
            )

            final_response = agent.run(
                run_config=RunConfig(
                    prompt=CodingCodeReviewPrompts.format_pr_review_plan_step(
                        diff_str=request.diff,
                    ),
                    system_prompt=CodingUnitTestPrompts.format_system_msg(),
                    model=AnthropicProvider.model("claude-3-5-sonnet-v2@20241022"),
                    run_name="Generate PR review",
                ),
            )

            if not final_response:
                return None

            # Get detailed PR description
            pr_description = llm_client.generate_structured(
                messages=agent.memory,
                prompt=CodingCodeReviewPrompts.format_pr_description_step(
                    diff_str=request.diff,
                ),
                model=GeminiProvider(model_name="gemini-2.0-flash-001"),
                response_format=CodePrReviewOutput.PrDescription,
                run_name="Generate PR description",
                max_tokens=4096,
            )

            formatted_response = llm_client.generate_structured(
                messages=agent.memory,
                prompt=CodingCodeReviewPrompts.pr_review_formatter_msg(),
                model=GeminiProvider(model_name="gemini-2.0-flash-001"),
                response_format=list[CodePrReviewOutput.Comment],
                run_name="Generate PR review structured",
                max_tokens=8192,
            )

            if not formatted_response or not formatted_response.parsed:
                return None

            for comment in formatted_response.parsed:
                if comment.suggestion:
                    comment.body += f"\n```suggestion\n{comment.suggestion}\n```"

            return CodePrReviewOutput(
                comments=formatted_response.parsed, description=pr_description.parsed
            )
