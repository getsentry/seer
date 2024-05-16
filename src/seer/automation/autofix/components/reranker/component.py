import sentry_sdk
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.reranker.models import (
    RawRerankerResult,
    RerankerOutput,
    RerankerRequest,
)
from seer.automation.autofix.components.reranker.prompts import RerankerPrompts
from seer.automation.autofix.utils import autofix_logger, escape_multi_xml
from seer.automation.component import BaseComponent


class RerankerComponent(BaseComponent[RerankerRequest, RerankerOutput]):
    context: AutofixContext

    @ai_track(description="Reranker")
    def invoke(self, request: RerankerRequest) -> RerankerOutput:
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

        with self.context.state.update() as cur:
            cur.usage += usage

        if not completion_result.content:
            autofix_logger.warning("Reranker agent did not return a valid response")
            return RerankerOutput(chunks=[])

        snippet_ids = RawRerankerResult.from_xml(
            f"<reranker_result>{escape_multi_xml(completion_result.content, ['thoughts'])}</reranker_result>"
        ).snippet_ids

        # Sanity log if we ever get a hash collision
        all_snippet_ids = [chunk.get_short_hash() for chunk in request.chunks]
        if len(all_snippet_ids) != len(set(all_snippet_ids)):
            sentry_sdk.capture_message(f"Hash collision in reranker: {all_snippet_ids}")

        relevant_chunks = []
        for short_snippet_hash in snippet_ids:
            chunk = next(
                chunk for chunk in request.chunks if chunk.matches_short_hash(short_snippet_hash)
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
