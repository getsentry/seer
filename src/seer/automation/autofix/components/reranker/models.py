import json
import textwrap

from pydantic import BaseModel
from pydantic_xml import element

from seer.automation.codebase.models import BaseDocumentChunk
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import PromptXmlModel


class RerankerRequest(BaseComponentRequest):
    query: str
    chunks: list[BaseDocumentChunk]


class RawRerankerResult(PromptXmlModel, tag="reranker_result"):
    thoughts: str = element()
    raw_snippet_ids: str = element(tag="code_snippet_ids")

    @property
    def snippet_ids(self) -> list[str]:
        return json.loads(self.raw_snippet_ids)


class RerankerOutput(BaseComponentOutput):
    chunks: list[BaseDocumentChunk]
