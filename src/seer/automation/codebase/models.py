import re
import textwrap
from functools import cached_property
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, model_serializer
from pydantic_xml import attr

from seer.automation.models import FilePatch, Hunk, PromptXmlModel, RepoDefinition, annotate_hunks


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
    previous_filename: str
    repo_full_name: str

    @cached_property
    def hunks(self) -> list[Hunk]:
        return FilePatch.to_hunks(self.patch)

    def overlapping_hunk_idxs(self, start_line: int, end_line: int | None = None) -> list[int]:
        if end_line is None:
            end_line = start_line
        hunk_ranges = [
            (hunk.target_start, hunk.target_start + hunk.target_length - 1) for hunk in self.hunks
        ]
        return [
            idx
            for idx, (hunk_start, hunk_end) in enumerate(hunk_ranges)
            if start_line <= hunk_end and hunk_start <= end_line
        ]

    def should_show_hunks(self) -> bool:
        if self.status == "removed":
            return False
        return self.changes > 0

    def format_hunks(self) -> str:
        return "\n\n".join(annotate_hunks(self.hunks))

    def format(self) -> str:
        tag_start = f"<file><filename>{self.filename}</filename>"

        if self.status == "renamed":
            title = f"File {self.previous_filename} was renamed to {self.filename}"
        elif self.status == "removed":
            title = f"File {self.filename} was removed"
        else:
            title = f"Here are the changes made to file {self.filename}"
        repo_name_addendum = f" in repo {self.repo_full_name}" if self.repo_full_name else ""
        title = title + repo_name_addendum

        if self.should_show_hunks():
            formatted_hunks = self.format_hunks()
        else:
            formatted_hunks = ""

        tag_end = "</file>"

        return "\n\n".join((tag_start, title, formatted_hunks, tag_end))


def format_diff(pr_files: list[PrFile]) -> str:
    body = "\n\n".join(pr_file.format() for pr_file in pr_files)
    return f"<diff>\n\n{body}\n\n</diff>"


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
    is_first_occurrence: bool
    rule_id: int | None = None
    rule: StaticAnalysisRule | None = None
    encoded_code_snippet: str | None = None
    potentially_related_issue_titles: list[str] | None = None

    def _try_get_language(self) -> str | None:
        if ".py" in self.encoded_location:
            return "python"
        if ".js" in self.encoded_location or ".ts" in self.encoded_location:
            return "javascript"
        if ".php" in self.encoded_location:
            return "php"
        return None

    @property
    def start_line(self) -> int:
        return int(Location.from_encoded(self.encoded_location).start_line)

    @property
    def end_line(self) -> int:
        return int(Location.from_encoded(self.encoded_location).end_line)

    def format_warning_id_and_message(self) -> str:
        return f"Warning (ID {self.id}): {self.message}"

    def format_warning(self, filename: str | None = None) -> str:
        location = Location.from_encoded(self.encoded_location)
        if not self.potentially_related_issue_titles:
            formatted_issue_titles = "    (no related issues found)"
        else:
            related_issue_titles = self.potentially_related_issue_titles or []
            formatted_issue_titles = "\n".join([f"* {title}" for title in related_issue_titles])

        if (language := self._try_get_language()) and (self.encoded_code_snippet):
            formatted_code_snippet = textwrap.dedent(
                """\
                ----------
                Code Snippet:
                ```{language}
                {snippet}
                ```
                """
            ).format(language=language, snippet=self.encoded_code_snippet)
        else:
            formatted_code_snippet = ""

        if self.is_first_occurrence:
            occurrence_message = "This is likely a newly introduced warning from this code change."
        else:
            occurrence_message = (
                "The warning at this location has been seen before prior to this code change."
            )

        return textwrap.dedent(
            """\
            <warning><warning_id>{id}</warning_id>
            {warning_id_and_message}
            ----------
            Location:
                filename: {location_filename}
                start_line: {location_start_line}
                end_line: {location_end_line}
            {formatted_code_snippet}----------
            Potentially related issue titles:
            {formatted_issue_titles}
            ----------
            Has this warning at this location been seen before?
                {occurrence_message}
            ----------
            {formatted_rule}</warning>"""
        ).format(
            id=self.id,
            warning_id_and_message=self.format_warning_id_and_message(),
            location_filename=filename or location.filename,
            location_start_line=location.start_line,
            location_end_line=location.end_line,
            formatted_code_snippet=formatted_code_snippet,
            formatted_issue_titles=formatted_issue_titles,
            formatted_rule=self.rule.format_rule() if self.rule else "",
            occurrence_message=occurrence_message,
        )
