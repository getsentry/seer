import json
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
    model_validator,
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
    in_app: bool | None = False
    vars: Optional[dict[str, Any]] = None
    package: Optional[str] = None

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
        # and filter out any values containing "[Filtered]"
        trimmed_vars = {}
        for key, val in vars.items():
            if key in code_context:
                if isinstance(val, (dict, list)):
                    filtered_val = StacktraceFrame._filter_nested_value(val)
                    if filtered_val is not None:
                        trimmed_vars[key] = filtered_val
                elif not StacktraceFrame._contains_filtered(val):
                    trimmed_vars[key] = val
        return trimmed_vars

    @staticmethod
    def _filter_nested_value(value: Any) -> Any:
        if isinstance(value, dict):
            filtered_dict = {}
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    filtered_v = StacktraceFrame._filter_nested_value(v)
                    if filtered_v is not None:
                        filtered_dict[k] = filtered_v
                elif not StacktraceFrame._contains_filtered(v):
                    filtered_dict[k] = v
            return filtered_dict if filtered_dict else None
        elif isinstance(value, list):
            filtered_list = []
            for item in value:
                if isinstance(item, (dict, list)):
                    filtered_item = StacktraceFrame._filter_nested_value(item)
                    if filtered_item is not None:
                        filtered_list.append(filtered_item)
                elif not StacktraceFrame._contains_filtered(item):
                    filtered_list.append(item)
            return filtered_list if filtered_list else None
        return None if StacktraceFrame._contains_filtered(value) else value

    @staticmethod
    def _contains_filtered(value: Any) -> bool:
        return isinstance(value, str) and "[Filtered]" in value


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

    def to_str(
        self,
        max_frames: int = 16,
        in_app_only: bool = False,
        include_context: bool = True,
        include_var_values: bool = True,
    ):
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

            if include_context:
                for ctx in frame.context:
                    is_suspect_line = ctx[0] == frame.line_no
                    stack_str += f"{ctx[1]}{'  <-- SUSPECT LINE' if is_suspect_line else ''}\n"

            if frame.vars:
                if include_var_values:
                    vars_title = "Variable values at the time of the exception:"
                    vars_str = json.dumps(frame.vars, indent=2)
                else:
                    vars_title = "Variables at the time of the exception:"
                    vars_str = ", ".join(frame.vars.keys())

                stack_str += textwrap.dedent(
                    """\
                    ---
                    {vars_title}:
                    {vars_str}
                    """
                ).format(vars_title=vars_title, vars_str=vars_str)
            stack_str += "------\n"

        return stack_str

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
    tags: NotRequired[list[dict[str, str | None]]]


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
    transaction_name: str | None = None
    exceptions: list[ExceptionDetails] = Field(default_factory=list, exclude=False)
    threads: list[ThreadDetails] = Field(default_factory=list, exclude=False)
    breadcrumbs: list[BreadcrumbsDetails] = Field(default_factory=list, exclude=False)

    @classmethod
    def from_event(cls, error_event: SentryEventData):
        MAX_THREADS = 8  # TODO: Smarter logic for max threads

        exceptions: list[ExceptionDetails] = []
        threads: list[ThreadDetails] = []
        breadcrumbs: list[BreadcrumbsDetails] = []
        transaction_name: str | None = None

        for tag in error_event.get("tags", []):
            if tag.get("key") == "transaction":
                transaction_name = tag.get("value")

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
                    # Skip breadcrumbs with filtered content in message or data
                    if StacktraceFrame._contains_filtered(
                        breadcrumb.get("message")
                    ) or StacktraceFrame._contains_filtered(str(breadcrumb.get("data"))):
                        continue
                    crumb_details = BreadcrumbsDetails.model_validate(breadcrumb)
                    breadcrumbs.append(crumb_details)

        return cls(
            title=error_event.get("title"),
            transaction_name=transaction_name,
            exceptions=exceptions,
            threads=threads,
            breadcrumbs=breadcrumbs,
        )

    def format_event(self):
        return textwrap.dedent(
            """\
            {title} {transaction}
            <exceptions>
            {exceptions}
            </exceptions>
            <breadcrumb_logs>
            {breadcrumbs}
            </breadcrumb_logs>
            """
        ).format(
            title=self.title,
            transaction=f"(occurred in: {self.transaction_name})" if self.transaction_name else "",
            exceptions=self.format_exceptions(),
            breadcrumbs=self.format_breadcrumbs(),
        )

    def format_event_without_breadcrumbs(
        self, include_context: bool = True, include_var_values: bool = True
    ):
        return textwrap.dedent(
            f"""\
            {self.title}
            <exceptions>
            {self.format_exceptions(include_context=include_context, include_var_values=include_var_values)}
            </exceptions>
            """
        )

    def format_exceptions(self, include_context: bool = True, include_var_values: bool = True):
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
                    exception.stacktrace.to_str(
                        include_context=include_context,
                        include_var_values=include_var_values,
                    )
                    if exception.stacktrace
                    else ""
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
                <breadcrumb_{i}{breadcrumb_type}{breadcrumb_category}{level}>
                {content}
                </breadcrumb_{i}>"""
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
    duration_ns: NotRequired[float]
    children: NotRequired[list["ProfileFrame"]]


class Profile(BaseModel):
    profile_matches_issue: bool = Field(default=False)
    execution_tree: list[ProfileFrame] = Field(default_factory=list)
    relevant_functions: set[str] = Field(default_factory=set)

    def format_profile(
        self,
        context_before: int = 20,
        context_after: int = 3,
    ) -> str:
        """
        Format the profile tree, focusing on relevant functions from the stacktrace.

        Args:
            context_before: Number of lines to include before first relevant function
            context_after: Number of lines to include after last relevant function

        Returns:
            str: Formatted profile string, showing relevant sections of the execution tree
        """
        full_profile = self._format_profile_helper(self.execution_tree)

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

    def _format_profile_helper(self, tree: list[ProfileFrame], prefix: str = "") -> str:
        """
        Returns a pretty-printed string representation of the execution tree using tree formatting.

        Args:
            tree: List of dictionaries representing the execution tree
            prefix: Current line prefix for tree structure

        Returns:
            str: Formatted string representation of the tree
        """
        if not tree:
            return ""

        result = []

        for i, node in enumerate(tree):
            is_last = i == len(tree) - 1

            # Format the current node
            function_name = node.get("function", "Unknown function")
            filename = node.get("filename", "Unknown file")
            location_info = f" ({filename}:{node.get('lineno', '')})"
            duration_ms = node.get("duration_ns", -1) / 1_000_000
            duration_info = f" - {duration_ms:.0f}ms" if duration_ms >= 0 else ""

            # Add tree structure characters
            if is_last:
                result.append(f"{prefix}└─ {function_name}{location_info}{duration_info}")
                child_prefix = f"{prefix}   "
            else:
                result.append(f"{prefix}├─ {function_name}{location_info}{duration_info}")
                child_prefix = f"{prefix}│  "

            # Recursively format children
            if node.get("children"):
                children_str = self._format_profile_helper(node.get("children", []), child_prefix)
                if children_str:
                    result.append(children_str)

        return "\n".join(result)


class Span(BaseModel):
    span_id: str
    title: str | None = None
    data: dict[str, Any] | None = None
    duration: str | None = None
    children: list["Span"] = Field(default_factory=list)


class TraceEvent(BaseModel):
    event_id: str | None = None
    title: str | None = None
    is_transaction: bool = False
    is_error: bool = False
    platform: str | None = None
    is_current_project: bool = True
    project_slug: str | None = None
    project_id: int | None = None
    duration: str | None = None
    profile_id: str | None = None
    children: list["TraceEvent"] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)

    def format_spans_tree(self) -> str:
        """
        Returns a formatted string representation of the span tree.
        """
        if not self.spans:
            return "No spans available"

        lines = [f"Spans for {self.title or 'Unnamed Event'}"]
        self._format_spans(self.spans, "", lines)
        return "\n".join(lines)

    def _format_spans(self, spans: list[Span], prefix: str, lines: list[str]) -> None:
        """
        Helper method to recursively format spans.

        Args:
            spans: List of spans to format
            prefix: Current prefix for tree structure
            lines: List of lines being built
        """
        if not spans:
            return

        # Group consecutive similar spans
        grouped_spans: list[list[Span]] = []
        current_group: list[Span] | None = None

        for span in spans:
            if current_group and self._are_spans_similar(current_group[0], span):
                current_group.append(span)
            else:
                if current_group:
                    grouped_spans.append(current_group)
                current_group = [span]

        if current_group:
            grouped_spans.append(current_group)

        # Format each group
        for i, group in enumerate(grouped_spans):
            is_last = i == len(grouped_spans) - 1
            span = group[0]
            count_suffix = f" (repeated {len(group)} times)" if len(group) > 1 else ""

            # Create the formatted span line
            span_line = self._format_span_line(span) + count_suffix

            # Add the appropriate prefix based on position in tree
            if is_last:
                lines.append(f"{prefix}└─ {span_line}")
                child_prefix = f"{prefix}   "
                data_prefix = f"{prefix}   "
            else:
                lines.append(f"{prefix}├─ {span_line}")
                child_prefix = f"{prefix}│  "
                data_prefix = f"{prefix}│  "

            # Add the span data as JSON if available
            if span.data:
                data_str = json.dumps(span.data, indent=2)
                data_lines = data_str.split("\n")
                for data_line in data_lines:
                    lines.append(f"{data_prefix} {data_line}")

            # Process children
            if span.children:
                self._format_spans(span.children, child_prefix, lines)

    def _format_span_line(self, span: Span) -> str:
        """Format a single span line."""
        parts = []

        title = span.title or "Unnamed Span"
        parts.append(title)
        if span.duration:
            parts.append(f"({span.duration})")

        return " ".join(parts)

    def _are_spans_similar(self, span1: Span, span2: Span) -> bool:
        """Check if two spans are similar enough to be grouped together."""
        if span1.title != span2.title:
            return False

        # Check if children structures are the same
        if len(span1.children) != len(span2.children):
            return False

        # If they have children, we consider them similar only if all children are similar
        for i in range(len(span1.children)):
            if not self._are_spans_similar(span1.children[i], span2.children[i]):
                return False

        return True


class TraceTree(BaseModel):
    trace_id: str | None = None
    org_id: int | None = None
    events: list[TraceEvent] = Field(default_factory=list)  # only expecting transactions and errors

    def format_trace_tree(self):
        if not self.events:
            return "Trace (empty)"

        lines = ["Trace"]
        self._format_events(self.events, "", lines, is_last_child=True)
        return "\n".join(lines)

    def _format_events(
        self, events: list[TraceEvent], prefix: str, lines: list[str], is_last_child: bool
    ):
        if not events:
            return

        # Group consecutive similar events
        grouped_events: list[list[TraceEvent]] = []
        current_group: list[TraceEvent] | None = None

        for event in events:
            if current_group and self._are_events_similar(current_group[0], event):
                current_group.append(event)
            else:
                if current_group:
                    grouped_events.append(current_group)
                current_group = [event]

        if current_group:
            grouped_events.append(current_group)

        # Format each group
        for i, group in enumerate(grouped_events):
            is_last = i == len(grouped_events) - 1
            event = group[0]
            count_suffix = f" (repeated {len(group)} times)" if len(group) > 1 else ""

            # Create the formatted event line
            event_line = self._format_event_line(event) + count_suffix

            # Add the appropriate prefix based on position in tree
            if is_last:
                lines.append(f"{prefix}└─ {event_line}")
                child_prefix = f"{prefix}   "
            else:
                lines.append(f"{prefix}├─ {event_line}")
                child_prefix = f"{prefix}│  "

            # Process children
            if event.children:
                self._format_events(event.children, child_prefix, lines, is_last)

    def _format_event_line(self, event: TraceEvent) -> str:
        parts = []

        # Add ERROR prefix if not a transaction
        prefix = "ERROR: " if event.is_error else ""

        # Add title
        title = event.title or "Unnamed Event"
        parts.append(f"{prefix}{title}")

        # Add duration if it exists
        if event.duration:
            parts.append(f"({event.duration})")

        # Add event_id (first 7 digits)
        if event.event_id:
            parts.append(f"(event ID: {event.event_id[:7]})")

        # Add project
        if event.project_slug and event.project_id:
            project_str = f"(project: {event.project_slug})"
            parts.append(project_str)

        # Add platform
        if event.platform:
            parts.append(f"({event.platform})")

        # Add profile
        if event.profile_id:
            parts.append("(profile available)")

        return " ".join(parts)

    def _are_events_similar(self, event1: TraceEvent, event2: TraceEvent) -> bool:
        """Check if two events are similar enough to be grouped together"""
        if event1.title != event2.title:
            return False

        # Check if children structures are the same
        if len(event1.children) != len(event2.children):
            return False

        # If they have children, we consider them similar only if all children match
        # This is a simplified check - for a full check we'd need to recursively compare children
        for i in range(len(event1.children)):
            if not self._are_events_similar(event1.children[i], event2.children[i]):
                return False

        return True

    def get_full_event_id(self, truncated_id: str) -> str | None:
        """Return the full event_id given the first 7 characters"""
        for event in self._get_all_events():
            if event.event_id and event.event_id.startswith(truncated_id):
                return event.event_id
        return None

    def get_event_by_id(self, truncated_id: str) -> TraceEvent | None:
        """Return the full TraceEvent object given a truncated event ID"""
        for event in self._get_all_events():
            if event.event_id and event.event_id.startswith(truncated_id):
                return event
        return None

    def _get_all_events(self) -> list[TraceEvent]:
        """Return a flattened list of all events in the tree"""
        all_events = []

        def collect_events(events):
            for event in events:
                all_events.append(event)
                if event.children:
                    collect_events(event.children)

        collect_events(self.events)
        return all_events


class RepoDefinition(BaseModel):
    provider: Annotated[str, Examples(("github", "integrations:github"))]
    owner: str
    name: str
    external_id: Annotated[str, Examples(specialized.ascii_words)]
    base_commit_sha: Optional[str] = None
    provider_raw: Optional[str] = None

    @property
    def full_name(self):
        return f"{self.owner}/{self.name}"

    @model_validator(mode="before")
    @classmethod
    def store_provider_raw(cls, data):
        if isinstance(data, dict) and "provider" in data and "provider_raw" not in data:
            data["provider_raw"] = data["provider"]
        return data

    @field_validator("provider", mode="after")
    @classmethod
    def validate_provider(cls, provider: str):
        return process_repo_provider(provider)

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
