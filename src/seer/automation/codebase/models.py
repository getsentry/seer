import textwrap
from typing import Optional

import numpy as np
from pydantic import BaseModel

from seer.db import DbDocumentChunk, DbRepositoryInfo


class Document(BaseModel):
    path: str
    text: str
    repo_id: int
    language: str


class DocumentChunk(BaseModel):
    id: Optional[int] = None
    content: str
    context: Optional[str]
    language: str
    hash: str
    path: str
    index: int
    first_line_number: int
    last_line_number: int
    token_count: int
    repo_id: int

    def get_dump_for_embedding(self):
        return """{context}{content}""".format(
            context=self.context if self.context else "",
            content=self.content,
        )

    def get_dump_for_llm(self, repo_name: str):
        return textwrap.dedent(
            """\
            [Lines {first_line_number}-{last_line_number} in "{path}" in repo "{repo_name}"]
            {context}{content}"""
        ).format(
            path=self.path,
            first_line_number=self.first_line_number,
            last_line_number=self.last_line_number,
            repo_name=repo_name,
            context=self.context if self.context else "",
            content=self.content,
        )

    def __str__(self):
        return textwrap.dedent(
            """\
            [Lines {first_line_number}-{last_line_number} in "{path}"]
            {context}{content}"""
        ).format(
            path=self.path,
            first_line_number=self.first_line_number,
            last_line_number=self.last_line_number,
            context=self.context if self.context else "",
            content=self.content,
        )

    def __repr__(self):
        return self.__str__()


class DocumentChunkWithEmbedding(DocumentChunk):
    embedding: np.ndarray

    def to_db_model(self) -> DbDocumentChunk:
        return DbDocumentChunk(
            id=self.id,
            repo_id=self.repo_id,
            path=self.path,
            index=self.index,
            hash=self.hash,
            token_count=self.token_count,
            first_line_number=self.first_line_number,
            last_line_number=self.last_line_number,
            embedding=self.embedding,
            language=self.language,
        )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()  # Convert ndarray to list for serialization
        }


class DocumentChunkWithEmbeddingAndId(DocumentChunkWithEmbedding):
    id: int


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
