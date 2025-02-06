import json
import re
import textwrap
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, model_serializer
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


# Copied from https://github.com/codecov/bug-prediction-research/blob/main/src/core/typings.py
class Location(BaseModel):
    filename: str
    start_line: str
    end_line: str

    def model_post_init(self, __context: Any) -> None:
        if not self.end_line:
            self.end_line = self.start_line

    @classmethod
    def from_encoded(cls, data: str):
        location_parts = re.match(
            r"(?P<filename>.+):(?P<start_line>\d+)(?:~(?P<end_line>\d+))?",
            data,
        )
        if not location_parts:
            msg = f"Invalid location encoding: {data}"
            raise ValueError(msg)
        grouped_dict = location_parts.groupdict()
        return cls(
            filename=grouped_dict["filename"],
            start_line=grouped_dict["start_line"],
            end_line=grouped_dict.get("end_line") or grouped_dict["start_line"],
        )

    @model_serializer
    def serialize_location(self) -> str:
        return self.encode()

    def encode(self) -> str:
        base = f"{self.filename}:{self.start_line}"
        if self.end_line != self.start_line:
            base += f"~{self.end_line}"
        return base


# Mostly copied from https://github.com/codecov/bug-prediction-research/blob/main/src/core/database/models.py
class SentryIssue(BaseModel):
    group_id: str
    commit_id: str
    title: str
    json_encoded_stacktraces: list[dict]
    error_location: str
    encoded_error_snippet: str | None
    encoded_local_context: str | None
    encoded_non_local_context: str | None
    project_name: str

    def format_error(self) -> str:
        return textwrap.dedent(
            f"""\
            Issue: {self.title}
            ----------
            Location:
            {self.error_location}
            ----------
            Error Snippet:
            {self.encoded_error_snippet}
            ----------
            Local Context:
            {self.encoded_local_context}
            ----------
            Non-Local Context:
            {self.encoded_non_local_context}
            ----------
            Stacktrace:
            {json.dumps(self.json_encoded_stacktraces, indent=2)}
            """
        )


class StaticAnalysisRule(BaseModel):
    id: int
    code: str
    tool: str
    is_autofixable: bool | None  # refers to "Quick fix" not seer autofix
    is_stable: bool | None
    category: str


class StaticAnalysisWarning(BaseModel):
    id: int
    commit_id: str
    code: str
    message: str
    encoded_location: str  # TODO: this is a predicate type
    encoded_code_snippet: str | None
    rule_id: int | None = None
    rule: StaticAnalysisRule | None = None
    # TODO: project info necessary for seer?

    def format_warning(self) -> str:
        location = Location.from_encoded(self.encoded_location)
        return textwrap.dedent(
            f"""\
            Warning message: {self.message}
            ----------
            Location:
                filename: {location.filename}
                start_line: {location.start_line}
                end_line: {location.end_line}
            ----------
            """
            + ("Rule:\n\t" + self.rule.model_dump_json(indent=2) if self.rule else "")
        )
