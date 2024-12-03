import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient
from seer.automation.autofix.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    CodePrReviewOutput,
    CodePrReviewRequest,
    CodegenPrReviewRequest,
    CodegenPrReviewResponse,
)
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
                    model=AnthropicProvider.model("claude-3-5-sonnet@20240620"),
                    run_name="Generate PR review",
                ),
            )

            print(final_response)
            return
            # take this response and generate comments
            # if no response, post comment to PR saying no suggestions were made
