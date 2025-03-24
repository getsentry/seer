import json
import logging

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import AnthropicProvider, LlmClient
from seer.automation.autofix.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import CodePrReviewOutput, CodePrReviewRequest
from seer.automation.codegen.prompts import CodingCodeReviewPrompts, CodingUnitTestPrompts
from seer.automation.component import BaseComponent
from seer.automation.utils import extract_text_inside_tags
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

            return self._format_output(final_response)

    @staticmethod
    def _format_output(llm_output: str) -> CodePrReviewOutput | None:
        comments_json = extract_text_inside_tags(llm_output, "comments")
        if len(comments_json) == 0:
            raise ValueError("Failed to extract pr review comments from LLM response")

        try:
            comments_data = json.loads(comments_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format inside <comments>: {comments_json}. Error: {e}")

        if not isinstance(comments_data, list):
            raise ValueError("Expected a list of JSON objects inside <comments>")

        results = []
        for comment_data in comments_data:
            # skip any llm output item if missing required keys
            expected_keys = list(CodePrReviewOutput.Comment.model_fields.keys())
            if not all(key in comment_data for key in expected_keys):
                continue

            body = comment_data["body"]
            if "code_suggestion" in comment_data:
                code_suggestion = comment_data["code_suggestion"]
                body += f"\n```suggestion\n{code_suggestion}\n```"
            results.append(
                CodePrReviewOutput.Comment(
                    path=comment_data["path"],
                    line=comment_data["line"],
                    body=body,
                    start_line=comment_data["start_line"],
                )
            )

        return CodePrReviewOutput(comments=results)
