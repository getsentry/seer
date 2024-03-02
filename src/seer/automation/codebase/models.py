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
    content: str
    context: Optional[str]
    language: str
    hash: str
    path: str
    index: int
    token_count: int
    repo_id: int

    @property
    def identity_tuple(self) -> tuple[int, str, int]:
        """
        Unique *within* a namespace, but not *across* them.  See similar method on DbDocumentChunk
        """
        return self.repo_id, self.path, self.index

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


class DocumentChunkWithEmbedding(DocumentChunk):
    embedding: np.ndarray

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
    sha: str

    @property
    def is_indexed(self) -> bool:
        return bool(self.sha)

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
