from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel
from pydantic_xml import attr

from seer.automation.models import PromptXmlModel, RepoDefinition


class DocumentPromptXml(PromptXmlModel, tag="document", skip_empty=True):
    path: str = attr()
    repository: str | None = attr(default=None)
    content: str


class BaseDocument(BaseModel):
    path: str
    text: str

    def get_prompt_xml(self, repo_name: str | None) -> DocumentPromptXml:
        return DocumentPromptXml(path=self.path, repository=repo_name, content=self.text)


class Document(BaseDocument):
    language: str


class RepoAccessCheckRequest(BaseModel):
    repo: RepoDefinition


class RepoAccessCheckResponse(BaseModel):
    has_access: bool


class MatchXml(PromptXmlModel, tag="result"):
    path: str = attr()
    context: str


class Match(BaseModel):
    line_number: int
    context: str


class SearchResult(BaseModel):
    relative_path: str
    matches: list[Match]
    score: float


class GithubPrReviewComment(TypedDict):
    commit_id: str
    body: str
    path: str
    side: NotRequired[Literal["LEFT", "RIGHT"]]
    line: NotRequired[int]
    start_line: NotRequired[int]
    start_side: NotRequired[Literal["LEFT", "RIGHT"]]
    in_reply_to: NotRequired[str]
