import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.models import DocumentChunkPromptXml, QueryResultDocumentChunk
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.automation.models import PromptXmlModel
from seer.automation.utils import get_autofix_client_and_agent


class RetrieverRequest(BaseComponentRequest):
    text: str
    intent: str | None = None
    top_k: int = 8
    include_short_hash_as_id: bool = False


class RetrieverOutputPromptXml(PromptXmlModel, tag="chunks"):
    chunks: list[DocumentChunkPromptXml]


class RetrieverOutput(BaseComponentOutput):
    chunks: list[QueryResultDocumentChunk]

    def to_xml(self) -> RetrieverOutputPromptXml:
        return RetrieverOutputPromptXml(chunks=[chunk.get_prompt_xml() for chunk in self.chunks])


class RetrieverPrompts:
    @staticmethod
    def format_plan_item_query_system_msg():
        return textwrap.dedent(
            """\
            Given the below query, please output a JSON with the "queries" field being an array of strings of multiple relevant queries that you would use to find the code that would satisfy the original query.

            ## Guidelines ##
            - The queries should be specific to the codebase and should be able to be used to find the code that would satisfy the original query.
            - The queries can be both keywords and semantic queries.
            - These queries will be used for an embedding cosine similarity search to find the relevant code.
            - The queries will match with code snippets that also include the file path.

            Examples are provided below:

            Original Query:
            "def get_abc("
            Intent:
            "Renaming the function `get_abc()` to `get_xyz()` in `static/app.py`."
            Improved Queries:
            {"queries": ["def get_abc", "get_abc", "static/app.py", "get_abc in static/app.py", "abc"]}

            Original Query:
            "/api/v1/health"
            Intent:
            "Find where the endpoint for `/api/v1/health` is defined in the codebase."
            Improved Queries:
            {"queries": ["/api/v1/health", "api/v1/health", "health", "health check", "liveliness", "ready"]}"""
        )

    @staticmethod
    def format_plan_item_query_default_msg(text: str, intent: str | None = None):
        return textwrap.dedent(
            """\
            Original Query:
            "{text}"{intent_msg}
            Improved Queries:
            """
        ).format(
            text=text, intent_msg='\nIntent:\n"{intent}"'.format(intent=intent) if intent else ""
        )


class RetrieverComponent(BaseComponent[RetrieverRequest, RetrieverOutput]):
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @observe(name="Retriever")
    @ai_track(description="Retriever")
    def invoke(self, request: RetrieverRequest) -> RetrieverOutput | None:
        # Identify good search queries for the plan item
        data, message, usage = get_autofix_client_and_agent()[0]().json_completion(
            messages=[
                Message(
                    role="user",
                    content=RetrieverPrompts.format_plan_item_query_default_msg(
                        text=request.text, intent=request.intent
                    ),
                ),
            ],
            system_prompt=RetrieverPrompts.format_plan_item_query_system_msg(),
        )

        with self.context.state.update() as cur:
            cur.usage += usage

        if data is None or "queries" not in data:
            autofix_logger.warning(f"No search queries found for instruction: '{request.text}'")
            return None

        queries = data["queries"]
        autofix_logger.debug(f"Search queries: {queries}")
        self.context.event_manager.add_log(f"Searching with queries: {queries}")

        unique_chunks: dict[str, QueryResultDocumentChunk] = {}
        for query in queries:
            retrived_chunks = self.context.query_all_codebases(query, top_k=request.top_k)
            for chunk in retrived_chunks:
                unique_chunks[chunk.hash] = chunk
        chunks = list(unique_chunks.values())

        autofix_logger.debug(f"Retrieved {len(chunks)} unique chunks.")

        return RetrieverOutput(chunks=chunks)
