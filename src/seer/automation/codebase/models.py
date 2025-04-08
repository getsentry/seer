import re
import textwrap
from functools import cached_property
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, model_serializer
from pydantic_xml import attr

from seer.automation.models import FilePatch, Hunk, PromptXmlModel, RepoDefinition


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
    repo_name: str = attr()
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


class PrFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    patch: str
    status: Literal["added", "removed", "modified", "renamed", "copied", "changed", "unchanged"]
    changes: int
    sha: str

    @cached_property
    def hunks(self) -> list[Hunk]:
        return FilePatch.to_hunks(self.patch)


# Mostly copied from https://github.com/codecov/bug-prediction-research/blob/main/src/core/database/models.py
class StaticAnalysisRule(BaseModel):
    id: int
    code: str
    tool: str
    is_autofixable: bool | None  # refers to "Quick fix"
    is_stable: bool | None
    category: str

    def format_rule(self) -> str:
        return textwrap.dedent(
            f"""\
            Static Analysis Rule:
                Rule: {self.code}
                Tool: {self.tool}
                Is auto-fixable: {self.is_autofixable}
                Is stable: {self.is_stable}
                Category: {self.category}
            """
        )


class StaticAnalysisWarning(BaseModel):
    id: int
    code: str
    message: str
    encoded_location: str
    rule_id: int | None = None
    rule: StaticAnalysisRule | None = None
    encoded_code_snippet: str | None = None

    def _try_get_language(self) -> str | None:
        if ".py" in self.encoded_location:
            return "python"
        if ".js" in self.encoded_location or ".ts" in self.encoded_location:
            return "javascript"
        if ".php" in self.encoded_location:
            return "php"
        return None

    def format_warning(self) -> str:
        location = Location.from_encoded(self.encoded_location)
        return (
            textwrap.dedent(
                f"""\
            Warning message: {self.message}
            ----------
            Location:
                filename: {location.filename}
                start_line: {location.start_line}
                end_line: {location.end_line}
            Code Snippet:
            ```{self._try_get_language() or ""}
            CODE_SNIPPET
            ```
            ----------
            FORMATTED_RULE
            """
            )
            # Multiline strings being substituted inside textwrap.dedent would mess with the formatting.
            # So we substitute afterwards.
            .replace(
                "CODE_SNIPPET", textwrap.dedent(self.encoded_code_snippet or "").strip()
            ).replace("FORMATTED_RULE", self.rule.format_rule() if self.rule else "")
        )
