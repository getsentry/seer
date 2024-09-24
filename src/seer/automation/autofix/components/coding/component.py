import logging

from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.agent import AgentConfig, ClaudeAgent
from seer.automation.agent.client import ClaudeClient, GptClient
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.coding.models import (
    ApplyResult,
    CodingOutput,
    CodingRequest,
    FileContext,
    FixStep,
    FixSteps,
    RootCausePlanTaskPromptXml,
)
from seer.automation.autofix.components.coding.prompts import CodingPrompts
from seer.automation.autofix.components.coding.utils import extract_diff_chunks
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.tools import BaseTools
from seer.automation.component import BaseComponent
from seer.automation.models import FileChange
from seer.automation.utils import extract_parsed_model
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class CodingComponent(BaseComponent[CodingRequest, CodingOutput]):
    context: AutofixContext

    def _append_file_change(self, repo_external_id: str, file_change: FileChange):
        with self.context.state.update() as cur:
            cur.codebases[repo_external_id].file_changes.append(file_change)

    @observe(name="Apply Diff")
    @inject
    def apply_diff(
        self, step: FixStep, claude_client: ClaudeClient = injected, max_attempts: int = 12
    ):
        file_contents = self.context.get_repo_client(step.repo_name).get_file_content(
            step.file_path
        )

        if not file_contents:
            return ApplyResult(
                success=False,
                reason="File not found",
                file_path=step.file_path,
                repo_name=step.repo_name,
                final_diff=None,
            )

        last_correct_content = ""
        accumulated_content = ""

        def check_stream(content: str) -> bool:
            if "@@" in content:
                diff_chunks = extract_diff_chunks(content)
                for chunk in diff_chunks:
                    if chunk.original_chunk not in file_contents:
                        return False
            return True

        @observe(name="Claude diff attempt")
        def attempt():
            nonlocal last_correct_content
            nonlocal accumulated_content

            model = "claude-3-5-sonnet@20240620"
            stream = claude_client.stream(
                messages=[
                    Message(
                        role="user",
                        content=CodingPrompts.format_apply_prompt(
                            step.file_path, step.diff, file_contents
                        ),
                    ),
                    Message(
                        role="assistant",
                        content=f"Here's the corrected unified diff with neighboring context lines and correct whitespace + indentations:\n\n<diff>{last_correct_content.rstrip()}",
                    ),
                ],
                model=model,
                max_tokens=2048,
                temperature=0.5,
                stop_sequences=["</diff>"],
            )

            for chunk in stream:
                if chunk.type == "text_delta":
                    accumulated_content += chunk.text
                    if not check_stream(accumulated_content):
                        print("Stream cancelled, retrying", accumulated_content)
                        accumulated_content = last_correct_content
                        if len(accumulated_content) > attempts * 16 and attempts > 2:
                            # Chuck off the end in chunks if we're still not converging
                            accumulated_content = accumulated_content[: -(attempts * 16)]
                            last_correct_content = accumulated_content
                            print(
                                "Chucked off the end, new accumulated content: ",
                                accumulated_content,
                            )
                        return None
                    last_correct_content = accumulated_content

            return ApplyResult(
                success=True,
                reason=None,
                file_path=step.file_path,
                repo_name=step.repo_name,
                final_diff=accumulated_content,
            )

        attempts = 0
        while attempts < max_attempts:
            result = attempt()
            if result:
                return result

            # If we're here, the stream was cancelled, so we'll increment attempts and continue
            attempts += 1

        # If we've reached this point, we've exceeded max_attempts
        return ApplyResult(
            success=False,
            reason=f"Failed after {max_attempts} attempts",
            file_path=step.file_path,
            repo_name=step.repo_name,
            final_diff=None,
        )

    def _get_file_contents_from_context(self, root_cause: RootCauseAnalysisItem):
        context_files: dict[str, FileContext] = dict()
        if isinstance(root_cause, RootCauseAnalysisItem) and root_cause.code_context:
            for context in root_cause.code_context:
                if context.snippet:
                    context_files[context.snippet.file_path] = FileContext(
                        file_path=context.snippet.file_path,
                        repo_name=context.snippet.repo_name,
                        content="",
                    )

        files_to_remove = set()
        for file_path, file_context in context_files.items():
            file_content = self.context.get_repo_client(file_context.repo_name).get_file_content(
                file_context.file_path
            )
            if file_content:
                file_context.content = file_content
            else:
                logger.warning(f"File not found: {file_path}")
                files_to_remove.add(file_path)

        for file_path in files_to_remove:
            context_files.pop(file_path)

        return "\n".join([item.to_prompt_str() for item in context_files.values()])

    @observe(name="Plan+Code")
    @ai_track(description="Plan+Code")
    @inject
    def invoke(self, request: CodingRequest, gpt_client: GptClient = injected) -> bool:
        tools = BaseTools(self.context)

        agent = ClaudeAgent(
            tools=tools.get_tools(),
            config=AgentConfig(system_prompt=CodingPrompts.format_system_msg(), interactive=False),
        )

        task_str = (
            RootCausePlanTaskPromptXml.from_root_cause(request.root_cause_and_fix).to_prompt_str()
            if isinstance(request.root_cause_and_fix, RootCauseAnalysisItem)
            else request.root_cause_and_fix
        )

        file_context_str = self._get_file_contents_from_context(request.root_cause_and_fix)

        state = self.context.state.get()

        response = agent.run(
            CodingPrompts.format_fix_discovery_msg(
                event=request.event_details.format_event(),
                task_str=task_str,
                summary=request.summary,
                file_context_str=file_context_str,
                repo_names=[repo.full_name for repo in state.request.repos],
                instruction=request.instruction,
            ),
            context=self.context,
        )

        if not response:
            return False

        complete = False
        retries = 0
        max_retries = 2
        while not complete and retries < max_retries:
            parsed_response = extract_parsed_model(
                gpt_client.openai_client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Given the following plan, extract out each step. Each step is a file change and should be a separate diff. Return the steps as a list of step items. In the 'original_text' field, include the entire original text content of the step instruction. ```\n{response}\n```",
                        }
                    ],
                    response_format=FixSteps,
                )
            )

            failed_steps = []
            repo_client = self.context.get_repo_client()
            for step in parsed_response.steps:
                apply_result = self.apply_diff(step)
                if not apply_result.success:
                    logger.warning(
                        f"Failed to apply diff for {step.file_path}: {apply_result.reason}"
                    )
                    failed_steps.append(step)

                if apply_result.final_diff:
                    diff_chunks = extract_diff_chunks(apply_result.final_diff)
                    for chunk in diff_chunks:
                        self._append_file_change(
                            repo_client.repo_external_id,
                            FileChange(
                                change_type="edit",
                                path=apply_result.file_path,
                                reference_snippet=chunk.original_chunk,
                                new_snippet=chunk.new_chunk,
                                description=apply_result.reason,
                                commit_message=apply_result.reason,
                            ),
                        )

            if not failed_steps:
                complete = True
            else:
                if retries >= max_retries:
                    raise Exception("Failed to apply all diffs")

                response = agent.run(
                    CodingPrompts.format_fix_retry_msg(
                        failed_steps=failed_steps,
                    ),
                    context=self.context,
                )

                if "<DONE>" in (response or ""):
                    complete = True

                retries += 1

        return True
