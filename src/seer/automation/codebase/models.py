import datetime
import hashlib
import textwrap
from typing import Annotated, Any, Mapping, Optional

import numpy as np
from johen import gen
from johen.examples import Examples
from johen.generators import specialized
from pydantic import BaseModel
from pydantic_xml import attr

from seer.automation.models import PromptXmlModel, RepoDefinition
from seer.db import DbCodebaseNamespace, DbRepositoryInfo


class CreateCodebaseTaskRequest(BaseModel):
    organization_id: int
    project_id: int
    repo: RepoDefinition


class UpdateCodebaseTaskRequest(BaseModel):
    repo_id: int


class Document(BaseModel):
    path: str
    text: str
    language: str


class DraftDocument(Document):
    text: str | None = None


lorem_ipsum_parts = "lLorem ipsum dolor sit amet, consectetur adipiscing elit.".split()
lorem_ipsum = (" ".join(r.choice(lorem_ipsum_parts) for _ in range(r.randint(15, 90))) for r in gen)

SHORT_HASH_LENGTH = 6


class DocumentChunkPromptXml(PromptXmlModel, tag="chunk", skip_empty=True):
    id: Optional[str] = attr(default=None)
    path: str = attr()
    repo: str = attr()
    content: str


class ChunkResult(BaseModel):
    hash: str
    path: str
    language: str
    index: int


class ChunkQueryResult(ChunkResult):
    distance: float


class BaseDocumentChunk(BaseModel):
    content: Annotated[str, Examples(lorem_ipsum)]
    context: Annotated[Optional[str], Examples(lorem_ipsum)]
    language: Annotated[str, Examples(("python",))]
    hash: Annotated[str, Examples(hashlib.sha1(s).hexdigest() for s in specialized.byte_strings)]
    path: Annotated[str, Examples(specialized.file_paths)]
    index: Annotated[int, Examples(specialized.ints)]
    token_count: Annotated[int, Examples(specialized.ints)]
    repo_name: Annotated[
        Optional[str], Examples(("getsentry/seer", "corps/johen", "getsentry/seer-automation"))
    ] = None

    def get_short_hash(self) -> str:
        return self.hash[:SHORT_HASH_LENGTH]

    def matches_short_hash(self, short_hash: str) -> bool:
        return self.get_short_hash() == short_hash

    def get_dump_for_embedding(self):
        return """{context}{content}""".format(
            context=self.context if self.context else "",
            content=self.content,
        )

    def get_dump_for_llm(self, include_short_hash_as_id: bool = False):
        xml_chunk = self.get_prompt_xml(include_short_hash_as_id)

        return xml_chunk.to_prompt_str()

    def get_prompt_xml(self, include_short_hash_as_id: bool = False):
        return DocumentChunkPromptXml(
            id=self.hash[:SHORT_HASH_LENGTH] if include_short_hash_as_id else None,
            path=self.path,
            repo=self.repo_name or "",
            content=(self.context if self.context else "") + self.content,
        )

    def get_db_metadata(self) -> Mapping[str, Any]:
        return dict(
            path=self.path,
            index=self.index,
            hash=self.hash,
            token_count=self.token_count,
            language=self.language,
        )

    def __str__(self):
        return textwrap.dedent(
            """\
            [{path}]
            {context}{content}"""
        ).format(
            path=self.path,
            context=self.context if self.context else "",
            content=self.content,
        )

    def __repr__(self):
        return self.__str__()


class QueryResultDocumentChunk(BaseDocumentChunk):
    distance: Annotated[float, Examples(r.randrange(0, 100, 1) / 100 for r in gen)]


class EmbeddedDocumentChunk(BaseDocumentChunk):
    embedding: Annotated[np.ndarray, Examples(np.full(768, r.random()) for r in gen)]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()  # Convert ndarray to list for serialization
        }


class RepositoryInfo(BaseModel):
    id: int
    organization: int
    project: int
    provider: str
    external_slug: str
    default_namespace: int

    @classmethod
    def from_db(cls, db_repo: DbRepositoryInfo) -> "RepositoryInfo":
        return cls(
            id=db_repo.id,
            organization=db_repo.organization,
            project=db_repo.project,
            provider=db_repo.provider,
            external_slug=db_repo.external_slug,
            default_namespace=db_repo.default_namespace,
        )

    def to_db_model(self) -> DbRepositoryInfo:
        return DbRepositoryInfo(
            id=self.id,
            organization=self.organization,
            project=self.project,
            provider=self.provider,
            external_slug=self.external_slug,
            default_namespace=self.default_namespace,
        )


class CodebaseNamespace(BaseModel):
    id: int
    repo_id: int
    sha: str
    tracking_branch: Optional[str]

    updated_at: datetime.datetime
    accessed_at: datetime.datetime

    @property
    def slug(self):
        if self.tracking_branch:
            return self.tracking_branch
        return self.sha

    @classmethod
    def from_db(cls, db_namespace: DbCodebaseNamespace) -> "CodebaseNamespace":
        return cls(
            id=db_namespace.id,
            repo_id=db_namespace.repo_id,
            sha=db_namespace.sha,
            tracking_branch=db_namespace.tracking_branch,
            updated_at=db_namespace.updated_at,
            accessed_at=db_namespace.accessed_at,
        )

    def to_db_model(self) -> DbCodebaseNamespace:
        return DbCodebaseNamespace(
            id=self.id,
            repo_id=self.repo_id,
            sha=self.sha,
            tracking_branch=self.tracking_branch,
            updated_at=self.updated_at,
            accessed_at=self.accessed_at,
        )
