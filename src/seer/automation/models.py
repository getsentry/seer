import json
import re
import textwrap
from typing import Annotated, Any, List, Literal, NotRequired, Optional
from xml.etree import ElementTree as ET

import sentry_sdk
from johen.examples import Examples
from johen.generators import specialized
from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic.alias_generators import to_camel, to_snake
from pydantic_xml import BaseXmlModel
from typing_extensions import TypedDict

from seer.automation.utils import process_repo_provider, unescape_xml_chars


class StacktraceFrame(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda k: AliasChoices(to_camel(k), to_snake(k)),
            serialization_alias=to_camel,
        )
    )

    function: Optional[Annotated[str, Examples(specialized.ascii_words)]] = None
    filename: Optional[Annotated[str, Examples(specialized.file_names)]]
    abs_path: Optional[Annotated[str, Examples(specialized.file_paths)]]
    line_no: Optional[int]
    col_no: Optional[int]
    context: list[tuple[int, Optional[str]]] = []
    repo_name: Optional[str] = None
    in_app: Annotated[bool, Examples((True, False))] = Field(default=False)
    vars: Optional[dict[str, Any]] = None
    package: Optional[str] = None

    @field_validator("in_app", mode="before")
    @classmethod
    def validate_in_app(cls, value: Any) -> bool:
        if value is None or not isinstance(value, bool):
            return False
        return value

    @field_validator("vars", mode="before")
    @classmethod
    def validate_vars(cls, vars: Optional[dict[str, Any]], info: ValidationInfo):
        if not vars or "context" not in info.data or not info.data["context"]:
            return vars
        code_str = ""
        for _, line in info.data["context"]:
            code_str += line + "\n"
        return cls._trim_vars(vars, code_str)

    @staticmethod
    def _trim_vars(vars: dict[str, Any], code_context: str):
        # only keep variables mentioned in the context of the stacktrace frame
        trimmed_vars = {}
        for key, val in vars.items():
            if key in code_context:
                trimmed_vars[key] = val
        return trimmed_vars


class SentryFrame(TypedDict):
    absPath: Optional[str]
    colNo: Optional[int]
    context: list[tuple[int, str]]
    filename: NotRequired[Optional[str]]
    function: NotRequired[Optional[str]]
    inApp: NotRequired[bool]
    instructionAddr: NotRequired[Optional[str]]
    lineNo: NotRequired[Optional[int]]
    module: NotRequired[Optional[str]]
    package: NotRequired[Optional[str]]
    platform: NotRequired[Optional[str]]
    rawFunction: NotRequired[Optional[str]]
    symbol: NotRequired[Optional[str]]
    symbolAddr: NotRequired[Optional[str]]
    trust: NotRequired[Optional[Any]]
    vars: NotRequired[Optional[dict[str, Any]]]
    addrMode: NotRequired[Optional[str]]
    isPrefix: NotRequired[bool]
    isSentinel: NotRequired[bool]
    lock: NotRequired[Optional[Any]]
    map: NotRequired[Optional[str]]
    mapUrl: NotRequired[Optional[str]]
    minGroupingLevel: NotRequired[int]
    origAbsPath: NotRequired[Optional[str]]
    sourceLink: NotRequired[Optional[str]]
    symbolicatorStatus: NotRequired[Optional[Any]]


class Stacktrace(BaseModel):
    frames: list[StacktraceFrame]

    @field_validator("frames", mode="before")
    @classmethod
    def validate_frames(cls, frames: list[StacktraceFrame | SentryFrame]):
        stacktrace_frames = []
        for frame in frames:
            if isinstance(frame, dict):
                if "function" not in frame:
                    frame["function"] = None
                try:
                    stacktrace_frames.append(StacktraceFrame.model_validate(frame))
                except ValidationError:
                    sentry_sdk.capture_exception()
                    continue
            else:
                stacktrace_frames.append(frame)

        return cls._trim_frames(stacktrace_frames)

    def to_str(self, max_frames: int = 16, in_app_only: bool = False):
        stack_str = ""

        frames = self.frames
        if in_app_only:
            frames = [frame for frame in frames if frame.in_app]

        for frame in reversed(frames[-max_frames:]):
            col_no_str = f", column {frame.col_no}" if frame.col_no is not None else ""
            repo_str = f" in repo {frame.repo_name}" if frame.repo_name else ""
            line_no_str = (
                f"[Line {frame.line_no}{col_no_str}]"
                if frame.line_no is not None
                else "[Line: Unknown]"
            )

            function = frame.function if frame.function else "Unknown function"
            if frame.filename:
                stack_str += f" {function} in file {frame.filename}{repo_str} {line_no_str} ({'In app' if frame.in_app else 'Not in app'})\n"
            elif frame.package:
                stack_str += f" {function} in package {frame.package} {line_no_str} ({'In app' if frame.in_app else 'Not in app'})\n"
            else:
                stack_str += f" {function} in unknown file {line_no_str} ({'In app' if frame.in_app else 'Not in app'})\n"

            for ctx in frame.context:
                is_suspect_line = ctx[0] == frame.line_no
                stack_str += f"{ctx[1]}{'  <-- SUSPECT LINE' if is_suspect_line else ''}\n"
            stack_str += (
                textwrap.dedent(
                    """\
                ---
                Variable values at the time of the exception:
                {vars_json_str}
                """
                ).format(vars_json_str=json.dumps(frame.vars, indent=2))
                if frame.vars
                else ""
            )
            stack_str += "------\n"

        stack_str = self._scrub_pii(stack_str)
        return stack_str

    def _scrub_pii(self, text: str) -> str:
        """
        Remove any personally identifiable identification from the given text.
        Not perfect, and is US/English-centric. Sometimes unintended strings get caught in the regex, too, so be careful.
        """
        checks = [
            (
                re.compile(
                    r"([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)",
                    re.IGNORECASE,
                ),
                "REDACTED_EMAIL",
            ),
            (
                re.compile(
                    r"((?:(?<![\d-])(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s])\d{3}[-.\s]\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s\d{2}\s\d{3}\s\d{4}(?![\d-])))"
                ),
                "REDACTED_PHONE_NUMBER",
            ),
            (
                re.compile(
                    r"((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))",
                    re.IGNORECASE,
                ),
                "REDACTED_PHONE_NUMBER",
            ),
            (
                re.compile("((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])"),
                "REDACTED_CREDIT_CARD",
            ),
            (
                re.compile(
                    r"\d{1,4} [\w\s]{1,20}(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)",
                    re.IGNORECASE,
                ),
                "REDACTED_STREET_ADDRESS",
            ),
            (re.compile(r"P\.? ?O\.? Box \d+", re.IGNORECASE), "REDACTED_PO_BOX"),
            (
                re.compile(
                    r"(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}"
                ),
                "REDACTED_SSN",
            ),
            (
                re.compile(r"[$]\s?[+-]?[0-9]{1,3}(?:(?:,?[0-9]{3}))*(?:\.[0-9]{1,2})?"),
                "REDACTED_PRICE",
            ),
            (
                re.compile(
                    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
                    re.IGNORECASE,
                ),
                "REDACTED_IP",
            ),
        ]
        for check in checks:
            pattern, replacement = check
            text = re.sub(pattern, replacement, text)
        return text

    @staticmethod
    def _trim_frames(frames: list[StacktraceFrame], frame_allowance=16):
        frames_len = len(frames)
        if frames_len <= frame_allowance:
            return frames

        app_frames = [frame for frame in frames if frame.in_app]
        system_frames = [frame for frame in frames if not frame.in_app]

        app_count = len(app_frames)
        system_allowance = max(frame_allowance - app_count, 0)
        app_allowance = frame_allowance - system_allowance

        if system_allowance > 0:
            # prioritize trimming system frames
            half_system = system_allowance // 2
            kept_system_frames = system_frames[:half_system] + system_frames[-half_system:]
        else:
            kept_system_frames = []

        if app_allowance > 0:
            half_app = app_allowance // 2
            kept_app_frames = app_frames[:half_app] + app_frames[-half_app:]
        else:
            kept_app_frames = []

        # combine and sort the kept frames based on their original order
        kept_frames = kept_system_frames + kept_app_frames
        kept_frames.sort(key=lambda frame: frames.index(frame))
        return kept_frames


class SentryStacktrace(TypedDict):
    frames: list[SentryFrame]


class SentryEventEntryDataValue(TypedDict):
    type: str
    value: str
    stacktrace: SentryStacktrace


class SentryExceptionEventData(TypedDict):
    values: list[SentryEventEntryDataValue]


class SentryExceptionEntry(BaseModel):
    type: Literal["exception"]
    data: SentryExceptionEventData


class SentryEventData(TypedDict):
    title: str
    entries: list[dict]


class ExceptionMechanism(TypedDict):
    type: str
    handled: NotRequired[bool]


class ExceptionDetails(BaseModel):
    type: Optional[str] = ""
    value: Optional[str] = ""
    stacktrace: Optional[Stacktrace] = None
    mechanism: Optional[ExceptionMechanism] = None

    @field_validator("stacktrace", mode="before")
    @classmethod
    def validate_stacktrace(cls, sentry_stacktrace: SentryStacktrace | Stacktrace):
        return (
            Stacktrace.model_validate(sentry_stacktrace)
            if isinstance(sentry_stacktrace, dict)
            else sentry_stacktrace
        )


class ThreadDetails(BaseModel):
    id: Optional[int | str] = None
    name: Optional[str] = None
    crashed: Optional[bool] = False
    current: Optional[bool] = False
    state: Optional[str] = None
    main: Optional[bool] = False

    stacktrace: Optional[Stacktrace] = None

    @field_validator("stacktrace", mode="before")
    @classmethod
    def validate_stacktrace(cls, sentry_stacktrace: SentryStacktrace | Stacktrace | None):
        return (
            Stacktrace.model_validate(sentry_stacktrace)
            if isinstance(sentry_stacktrace, dict)
            else sentry_stacktrace
        )


class BreadcrumbsDetails(BaseModel):
    type: Optional[str] = None
    message: Optional[str] = None
    category: Optional[str] = None
    data: Optional[dict] = None
    level: Optional[str] = None


class EventDetails(BaseModel):
    title: str
    exceptions: list[ExceptionDetails] = Field(default_factory=list, exclude=False)
    threads: list[ThreadDetails] = Field(default_factory=list, exclude=False)
    breadcrumbs: list[BreadcrumbsDetails] = Field(default_factory=list, exclude=False)

    @classmethod
    def from_event(cls, error_event: SentryEventData):
        MAX_THREADS = 8  # TODO: Smarter logic for max threads

        exceptions: list[ExceptionDetails] = []
        threads: list[ThreadDetails] = []
        breadcrumbs: list[BreadcrumbsDetails] = []
        for entry in error_event.get("entries", []):
            if entry.get("type") == "exception":
                for exception in entry.get("data", {}).get("values", []):
                    exceptions.append(ExceptionDetails.model_validate(exception))
            elif entry.get("type") == "threads":
                for thread in entry.get("data", {}).get("values", []):
                    thread_details = ThreadDetails.model_validate(thread)
                    if (
                        thread_details.stacktrace
                        and thread_details.stacktrace.frames
                        and len(threads) < MAX_THREADS
                    ):
                        threads.append(thread_details)
            elif entry.get("type") == "breadcrumbs":
                all_breadcrumbs = entry.get("data", {}).get("values", [])
                for breadcrumb in all_breadcrumbs[-10:]:  # only look at the most recent breadcrumbs
                    crumb_details = BreadcrumbsDetails.model_validate(breadcrumb)
                    if "[Filtered]" in (crumb_details.message or "") or "[Filtered]" in (
                        str(crumb_details.data) or ""
                    ):
                        continue
                    breadcrumbs.append(crumb_details)

        return cls(
            title=error_event.get("title"),
            exceptions=exceptions,
            threads=threads,
            breadcrumbs=breadcrumbs,
        )

    def format_event(self):
        return textwrap.dedent(
            """\
            {title}
            Exceptions:
            {exceptions}
            ----------
            Event Logs:
            {breadcrumbs}
            ----------
            """
        ).format(
            title=self.title,
            exceptions=self.format_exceptions(),
            breadcrumbs=self.format_breadcrumbs(),
        )

    def format_exceptions(self):
        return "\n".join(
            textwrap.dedent(
                """\
                    <exception_{i}{handled}{exception_type}{exception_message}>
                    {stacktrace}
                    </exception{i}>"""
            ).format(
                i=i,
                exception_type=f' type="{exception.type}"' if exception.type else "",
                exception_message=f' message="{exception.value}"' if exception.value else "",
                stacktrace=(
                    exception.stacktrace.to_str(in_app_only=True) if exception.stacktrace else ""
                ),
                handled=(
                    f' is_exception_handled="{"yes" if exception.mechanism.get("handled") else "no"}"'
                    if exception.mechanism and exception.mechanism.get("handled", None) is not None
                    else ""
                ),
            )
            for i, exception in enumerate(self.exceptions)
        )

    def format_threads(self):
        return "\n".join(
            textwrap.dedent(
                """\
                    <thread_{thread_id} name="{thread_name}" is_current="{thread_current}" state="{thread_state}" is_main="{thread_main}" crashed="{thread_crashed}">
                    <stacktrace>
                    {stacktrace}
                    </stacktrace>
                    </thread_{thread_id}>"""
            ).format(
                thread_id=thread.id,
                thread_name=thread.name,
                thread_state=thread.state,
                thread_current=thread.current,
                thread_crashed=thread.crashed,
                thread_main=thread.main,
                stacktrace=thread.stacktrace.to_str() if thread.stacktrace else "",
            )
            for thread in self.threads
        )

    def format_breadcrumbs(self):
        return "\n".join(
            textwrap.dedent(
                """\
                <event_log_{i}{breadcrumb_type}{breadcrumb_category}{level}>
                {content}
                </event_log_{i}>"""
            ).format(
                i=i,
                breadcrumb_type=f' type="{breadcrumb.type}"' if breadcrumb.type else "",
                breadcrumb_category=(
                    f' category="{breadcrumb.category}"' if breadcrumb.category else ""
                ),
                content="\n".join(
                    filter(
                        None,
                        [
                            f"{breadcrumb.message}\n" if breadcrumb.message else "",
                            (
                                f"{str({k: v for k, v in breadcrumb.data.items() if v})}\n"
                                if breadcrumb.data
                                else ""
                            ),
                        ],
                    )
                ),
                level=f' level="{breadcrumb.level}"' if breadcrumb.level else "",
            )
            for i, breadcrumb in enumerate(self.breadcrumbs)
        )


class IssueDetails(BaseModel):
    id: Annotated[int, Examples(specialized.unsigned_ints)]
    title: Annotated[str, Examples(specialized.ascii_words)]
    short_id: Optional[str] = None
    events: list[SentryEventData]


class ProfileFrame(TypedDict):
    function: str
    module: str
    filename: str
    lineno: int
    in_app: bool
    children: NotRequired[list["ProfileFrame"]]


class Profile(BaseModel):
    profile_matches_issue: bool = Field(default=False)
    execution_tree: list[ProfileFrame] = Field(default_factory=list)
    relevant_functions: set[str] = Field(default_factory=set)

    def format_profile(
        self,
        indent: int = 0,
        context_before: int = 20,
        context_after: int = 3,
    ) -> str:
        """
        Format the profile tree, focusing on relevant functions from the stacktrace.

        Args:
            indent: Base indentation level for the tree
            context_before: Number of lines to include before first relevant function
            context_after: Number of lines to include after last relevant function

        Returns:
            str: Formatted profile string, showing relevant sections of the execution tree
        """
        full_profile = self._format_profile_helper(self.execution_tree, indent)

        if self.relevant_functions:
            relevant_window = self._get_relevant_code_window(
                full_profile, context_before=context_before, context_after=context_after
            )
            if relevant_window:
                return relevant_window

        return full_profile

    def _get_relevant_code_window(
        self, code: str, context_before: int = 20, context_after: int = 3
    ) -> str | None:
        """
        Find the relevant section of code containing functions from the stacktrace.
        Expands the selection to include context lines before and after.

        Args:
            code: Multi-line string of formatted profile to analyze
            context_before: Number of lines to include before first relevant line
            context_after: Number of lines to include after last relevant line

        Returns:
            str | None: Selected profile window with context, or None if no relevant functions found
        """
        if not self.relevant_functions or not code:
            return None

        lines = code.splitlines()
        first_relevant_line = None
        last_relevant_line = None

        # Find first and last lines containing relevant functions
        for i, line in enumerate(lines):
            if any(func in line for func in self.relevant_functions):
                if first_relevant_line is None:
                    first_relevant_line = i
                last_relevant_line = i

        if first_relevant_line is None:
            first_relevant_line = 0
        if last_relevant_line is None:
            last_relevant_line = len(lines) - 1

        # Calculate window boundaries with context
        start_line = max(0, first_relevant_line - context_before)
        end_line = min(len(lines), last_relevant_line + context_after + 1)

        result = []
        if start_line > 0:
            result.append("...")
        result.extend(lines[start_line:end_line])
        if end_line < len(lines):
            result.append("...")

        return "\n".join(result)

    def _format_profile_helper(self, tree: list[ProfileFrame], indent: int = 0) -> str:
        """
        Returns a pretty-printed string representation of the execution tree with indentation.

        Args:
            tree: List of dictionaries representing the execution tree
            indent: Current indentation level (default: 0)

        Returns:
            str: Formatted string representation of the tree
        """
        result = []

        for node in tree:
            indent_str = "  " * indent

            func_line = f"{indent_str}â†’ {node.get('function')}"
            location = f"{node.get('filename')}"
            func_line += f" ({location})"

            result.append(func_line)

            # Recursively format children with increased indentation
            if node.get("children"):
                result.append(self._format_profile_helper(node.get("children", []), indent + 1))

        return "\n".join(result)


class RepoDefinition(BaseModel):
    provider: Annotated[str, Examples(("github", "integrations:github"))]
    owner: str
    name: str
    external_id: Annotated[str, Examples(specialized.ascii_words)]
    base_commit_sha: Optional[str] = None

    @property
    def full_name(self):
        return f"{self.owner}/{self.name}"

    @field_validator("provider", mode="after")
    @classmethod
    def validate_provider(cls, provider: str):
        cleaned_provider = process_repo_provider(provider)

        if cleaned_provider != "github":
            raise ValueError(f"Provider {cleaned_provider} is not supported.")

        return cleaned_provider

    def __hash__(self):
        return hash((self.provider, self.owner, self.name, self.external_id))


class InitializationError(Exception):
    pass


class PromptXmlModel(BaseXmlModel):
    def _pad_with_newlines(self, tree: ET.Element) -> None:
        for elem in tree.iter():
            if elem.text:
                stripped = elem.text.strip("\n")
                if stripped:
                    elem.text = "\n" + stripped + "\n"
            if elem.tail:
                stripped = elem.tail.strip("\n")
                if stripped:
                    elem.tail = "\n" + stripped + "\n"

    def to_prompt_str(self) -> str:
        tree: ET.Element = self.to_xml_tree()

        ET.indent(tree, space="", level=0)

        self._pad_with_newlines(tree)

        return unescape_xml_chars(ET.tostring(tree, encoding="unicode"))


class Line(BaseModel):
    source_line_no: Optional[int] = None
    target_line_no: Optional[int] = None
    diff_line_no: Optional[int] = None
    value: str
    line_type: Literal[" ", "+", "-"]


class Hunk(BaseModel):
    source_start: int
    source_length: int
    target_start: int
    target_length: int
    section_header: str
    lines: List[Line]


class FilePatch(BaseModel):
    type: Literal["A", "M", "D"]
    path: str
    added: int
    removed: int
    source_file: str
    target_file: str
    hunks: List[Hunk]

    def apply(self, file_contents: str | None) -> str | None:
        if self.type == "A":
            if file_contents is not None and file_contents.strip():
                raise FileChangeError("Cannot add a file that already exists.")
            return self._apply_hunks([])

        if file_contents is None:
            raise FileChangeError("File contents must be provided for modify or delete operations.")

        if self.type == "D":
            return None

        # For M type
        try:
            new_contents = self._apply_hunks(file_contents.splitlines(keepends=True))
        except Exception as e:
            raise FileChangeError(f"Error applying hunks: {e}")

        # Preserve any trailing characters from original
        if file_contents:
            trailing = file_contents[len(file_contents.rstrip()) :]
            return new_contents + trailing

        return new_contents

    def _apply_hunks(self, lines: List[str]) -> str:
        result = []
        current_line = 0

        for hunk in self.hunks:
            # Add unchanged lines before the hunk
            result.extend(lines[current_line : hunk.source_start - 1])
            current_line = hunk.source_start - 1

            for line in hunk.lines:
                if line.line_type == "+":
                    result.append(line.value + ("\n" if not line.value.endswith("\n") else ""))
                elif line.line_type == " ":
                    result.append(lines[current_line])
                    current_line += 1
                elif line.line_type == "-":
                    current_line += 1

        # Add any remaining unchanged lines after the last hunk
        result.extend(lines[current_line:])

        return "".join(result).rstrip("\n")


class FileChangeError(Exception):
    pass


class FileChange(BaseModel):
    change_type: Literal["create", "edit", "delete"]
    path: str
    reference_snippet: Optional[str] = None
    new_snippet: Optional[str] = None
    description: Optional[str] = None
    commit_message: Optional[str] = None

    def apply(self, file_contents: str | None) -> str | None:
        if self.change_type == "create":
            if file_contents is not None and file_contents != "":
                raise FileChangeError("Cannot create a file that already exists.")
            if self.new_snippet is None:
                raise FileChangeError("New snippet must be provided for creating a file.")
            return self.new_snippet

        if file_contents is None:
            raise FileChangeError("File contents must be provided for non-create operations.")

        if self.change_type == "edit":
            if self.new_snippet is None:
                raise FileChangeError("New snippet must be provided for editing a file.")
            if self.reference_snippet is None:
                raise FileChangeError("Reference snippet must be provided for editing a file.")
            return file_contents.replace(self.reference_snippet, self.new_snippet)

        # Delete
        if self.reference_snippet is None:
            return None

        return file_contents.replace(self.reference_snippet, "")
