import hashlib
import textwrap
from typing import Annotated, Optional

import numpy as np
from johen import gen
from johen.examples import Examples
from johen.generators import specialized
from pydantic import BaseModel
from pydantic_xml import attr

from seer.automation.models import PromptXmlModel
from seer.db import DbDocumentChunk, DbRepositoryInfo


class Document(BaseModel):
    path: str
    text: str
    language: str


lorem_ipsum_parts = "lLorem ipsum dolor sit amet, consectetur adipiscing elit.".split()
lorem_ipsum = (" ".join(r.choice(lorem_ipsum_parts) for _ in range(r.randint(15, 90))) for r in gen)

SHORT_HASH_LENGTH = 6


class DocumentChunkPromptXml(PromptXmlModel, tag="chunk", skip_empty=True):
    id: Optional[str] = attr(default=None)
    path: str = attr()
    repo: str = attr()
    content: str


class BaseDocumentChunk(PromptXmlModel):
    id: Optional[int] = None
    content: Annotated[str, Examples(lorem_ipsum)]
    context: Optional[str]
    language: Annotated[str, Examples(("python",))]
    hash: Annotated[str, Examples(hashlib.sha1(s).hexdigest() for s in specialized.byte_strings)]
    path: Annotated[str, Examples(specialized.file_paths)]
    index: int
    token_count: int

    def matches_short_hash(self, short_hash: str) -> bool:
        return self.hash[:SHORT_HASH_LENGTH] == short_hash

    def get_dump_for_embedding(self):
        return """{context}{content}""".format(
            context=self.context if self.context else "",
            content=self.content,
        )

    def get_dump_for_llm(
        self, repo_name: str | None = None, include_short_hash_as_id: bool = False
    ):
        xml_chunk = self.get_prompt_xml(repo_name or "", include_short_hash_as_id)

        return xml_chunk.to_prompt_str()

    def get_prompt_xml(self, repo_name: str | None = None, include_short_hash_as_id: bool = False):
        return DocumentChunkPromptXml(
            id=self.hash[:SHORT_HASH_LENGTH] if include_short_hash_as_id else None,
            path=self.path,
            repo=repo_name or "",
            content=(self.context if self.context else "") + self.content,
        )

    def to_partial_db_model(self, repo_id: int, namespace: str | None = None) -> DbDocumentChunk:
        return DbDocumentChunk(
            id=self.id,
            repo_id=repo_id,
            path=self.path,
            index=self.index,
            hash=self.hash,
            token_count=self.token_count,
            language=self.language,
            namespace=namespace,
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


class EmbeddedDocumentChunk(BaseDocumentChunk):
    embedding: Annotated[np.ndarray, Examples(np.full(768, r.random()) for r in gen)]

    def to_db_model(self, repo_id: int, namespace: str | None = None) -> DbDocumentChunk:
        db_chunk = self.to_partial_db_model(repo_id, namespace)
        db_chunk.embedding = self.embedding
        return db_chunk

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()  # Convert ndarray to list for serialization
        }


class StoredDocumentChunk(EmbeddedDocumentChunk):
    id: int
    repo_id: int


class StoredDocumentChunkWithRepoName(StoredDocumentChunk):
    repo_name: str

    def get_dump_for_llm(
        self, repo_name: str | None = None, include_short_hash_as_id: bool = False
    ):
        return super().get_dump_for_llm(
            repo_name or self.repo_name, include_short_hash_as_id=include_short_hash_as_id
        )

    def get_prompt_xml(self, repo_name: str | None = None, include_short_hash_as_id: bool = False):
        return super().get_prompt_xml(
            repo_name or self.repo_name, include_short_hash_as_id=include_short_hash_as_id
        )

    def get_prompt_xml(self, include_short_hash_as_id: bool = False):
        return self._get_prompt_xml(
            self.repo_name, include_short_hash_as_id=include_short_hash_as_id
        )


class RepositoryInfo(BaseModel):
    id: int
    organization: int
    project: int
    provider: str
    external_slug: str
    sha: str

    @classmethod
    def from_db(cls, db_repo: DbRepositoryInfo) -> "RepositoryInfo":
        return cls(
            id=db_repo.id,
            organization=db_repo.organization,
            project=db_repo.project,
            provider=db_repo.provider,
            external_slug=db_repo.external_slug,
            sha=db_repo.sha,
        )
