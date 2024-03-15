import textwrap
from typing import Optional

import numpy as np
from pydantic import BaseModel

from seer.db import DbDocumentChunk, DbRepositoryInfo


class Document(BaseModel):
    path: str
    text: str
    language: str


class BaseDocumentChunk(BaseModel):
    id: Optional[int] = None
    content: str
    context: Optional[str]
    language: str
    hash: str
    path: str
    index: int
    token_count: int

    def get_dump_for_embedding(self):
        return """{context}{content}""".format(
            context=self.context if self.context else "",
            content=self.content,
        )

    def get_dump_for_llm(self, repo_name: str):
        return textwrap.dedent(
            """\
            ["{path}" in repo "{repo_name}"]
            {context}{content}"""
        ).format(
            path=self.path,
            repo_name=repo_name,
            context=self.context if self.context else "",
            content=self.content,
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
    embedding: np.ndarray

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
