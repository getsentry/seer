import textwrap

from langsmith import traceable

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.models import DocumentChunkPromptXml, QueryResultDocumentChunk
from seer.automation.component import BaseComponent, BaseComponentOutput, BaseComponentRequest
from seer.automation.models import PromptXmlModel


class RetrieverRequest(BaseComponentRequest):
    text: str
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
            Given the below instruction, please output a JSON with the "queries" field being an array of strings of queries that you would use to find the code that would accomplish the instruction.

            ## Guidelines ##
            - The queries should be specific to the codebase and should be able to be used to find the code that would accomplish the instruction.
            - The queries can be both keywords and semantic queries.

            Examples are provided below:

            Instruction:
            "Rename the function `get_abc()` to `get_xyz()` in `static/app.py`."
            Queries:
            {"queries": ["get_abc", "static/app.py", "get_abc in static/app.py"]}

            Instruction:
            "Find where the endpoint for `/api/v1/health` is defined in the codebase."
            Queries:
            {"queries": ["/api/v1/health", "health endpoint", "health api", "status check"]}"""
        )

    @staticmethod
    def format_plan_item_query_default_msg(text: str):
        return textwrap.dedent(
            """\
            Instruction:
            "{text}"
            Queries:
            """
        ).format(text=text)


class RetrieverComponent(BaseComponent[RetrieverRequest, RetrieverOutput]):
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @traceable(name="Retriever", run_type="retriever", tags=["retriever:v1.1"])
    def invoke(self, request: RetrieverRequest) -> RetrieverOutput | None:
        with self.context.state.update() as cur:
            # Identify good search queries for the plan item
            data, message, usage = GptClient().json_completion(
                messages=[
                    Message(
                        role="system", content=RetrieverPrompts.format_plan_item_query_system_msg()
                    ),
                    Message(
                        role="user",
                        content=RetrieverPrompts.format_plan_item_query_default_msg(
                            text=request.text
                        ),
                    ),
                ]
            )

            cur.usage += usage

            if data is None or "queries" not in data:
                autofix_logger.warning(f"No search queries found for instruction: '{request.text}'")
                return None

            queries = data["queries"]
            autofix_logger.debug(f"Search queries: {queries}")

            context_dump = ""
            unique_chunks: dict[str, QueryResultDocumentChunk] = {}
            for query in queries:
                retrived_chunks = self.context.query_all_codebases(query, top_k=request.top_k)
                for chunk in retrived_chunks:
                    unique_chunks[chunk.hash] = chunk
            chunks = list(unique_chunks.values())

            autofix_logger.debug(f"Retrieved {len(chunks)} unique chunks.")

            for chunk in chunks:
                context_dump += f"\n\n{chunk.get_dump_for_llm(include_short_hash_as_id=request.include_short_hash_as_id)}"

            return RetrieverOutput(chunks=chunks)
