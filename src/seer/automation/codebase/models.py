import textwrap
from typing import Optional

import numpy as np
from pydantic import BaseModel

from seer.db import DbDocumentChunk


class Document(BaseModel):
    path: str
    text: str


class DocumentChunk(BaseModel):
    id: Optional[int] = None
    content: str
    context: Optional[str]
    hash: str
    path: str
    index: int
    first_line_number: int
    token_count: int

    def get_dump_for_embedding(self):
        return """{context}{content}""".format(
            context=self.context if self.context else "",
            content=self.content,
        )

    def get_dump_for_llm(self):
        return self.__str__()

    def __str__(self):
        return textwrap.dedent(
            """\
            [Lines {first_line_number}-{last_line_number} in "{path}"]
            {context}{content}"""
        ).format(
            path=self.path,
            first_line_number=self.first_line_number,
            last_line_number=self.first_line_number + len(self.content.split("\n")),
            context=self.context if self.context else "",
            content=self.content,
        )

    def __repr__(self):
        return self.__str__()


class DocumentChunkWithEmbedding(DocumentChunk):
    embedding: np.ndarray

    def to_db_model(self, repo_id: int) -> DbDocumentChunk:
        return DbDocumentChunk(
            id=self.id,
            repository_id=repo_id,
            path=self.path,
            index=self.index,
            hash=self.hash,
            token_count=self.token_count,
            first_line_number=self.first_line_number,
            embedding=self.embedding,
        )

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


class UninitializedRepoInfo(RepositoryInfo):
    id: None = None
