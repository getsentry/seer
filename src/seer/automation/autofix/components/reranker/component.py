from langsmith import traceable

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.reranker.models import (
    RawRerankerResult,
    RerankerOutput,
    RerankerRequest,
)
from seer.automation.autofix.components.reranker.prompts import RerankerPrompts
from seer.automation.autofix.utils import autofix_logger
from seer.automation.component import BaseComponent


class RerankerComponent(BaseComponent[RerankerRequest, RerankerOutput]):
    context: AutofixContext

    @traceable(name="Reranker", run_type="llm", tags=["reranker:v1"])
    def invoke(self, request: RerankerRequest) -> RerankerOutput:
        with self.context.state.update() as cur:
            gpt_client = GptClient()

            code_dump = "\n".join(
                [chunk.get_dump_for_llm(include_short_hash_as_id=True) for chunk in request.chunks]
            )

            completion_result, usage = gpt_client.completion(
                [
                    Message(role="system", content=RerankerPrompts.format_system_msg()),
                    Message(
                        role="user",
                        content=RerankerPrompts.format_default_msg(request.query, code_dump),
                    ),
                ]
            )

            cur.usage += usage

            snippet_ids = RawRerankerResult.from_xml(
                f"<research_result>{completion_result.content}</research_result>"
            ).snippet_ids

            relevant_chunks = []
            for short_snippet_hash in snippet_ids:
                chunk = next(
                    chunk
                    for chunk in request.chunks
                    if chunk.matches_short_hash(short_snippet_hash)
                )

                if not chunk:
                    # Go forward, but we should log this.
                    autofix_logger.exception(
                        ValueError(
                            f"Snippet with hash {short_snippet_hash} not found in the input chunks"
                        )
                    )
                    continue

                relevant_chunks.append(chunk)

            return RerankerOutput(chunks=relevant_chunks)
