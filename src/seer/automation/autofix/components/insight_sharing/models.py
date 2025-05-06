from enum import Enum

from pydantic import BaseModel

from seer.automation.component import BaseComponentOutput
from seer.automation.models import FilePatch


class InsightSharingType(str, Enum):
    INSIGHT = "insight"
    FILE_CHANGE = "file_change"


class InsightSources(BaseModel):
    stacktrace_used: bool
    breadcrumbs_used: bool
    http_request_used: bool
    trace_event_ids_used: list[str]
    connected_error_ids_used: list[str]
    profile_ids_used: list[str]
    code_used_urls: list[str]
    diff_urls: list[str]
    event_trace_id: str | None = None
    thoughts: str


class InsightSharingOutput(BaseComponentOutput):
    insight: str
    justification: str = ""
    markdown_snippets: str = ""
    change_diff: list[FilePatch] | None = None
    generated_at_memory_index: int = -1
    type: InsightSharingType = InsightSharingType.INSIGHT
    sources: InsightSources | None = None


class StacktraceSource(BaseModel):
    stacktrace_used: bool


class BreadcrumbsSource(BaseModel):
    breadcrumbs_used: bool


class HttpRequestSource(BaseModel):
    http_request_used: bool


class TraceEventSource(BaseModel):
    trace_event_id: str


class ConnectedErrorSource(BaseModel):
    connected_error_id: str


class ProfileSource(BaseModel):
    trace_event_id: str


class CodeSource(BaseModel):
    file_name: str
    repo_name: str
    code_snippet: str


class DiffSource(BaseModel):
    repo_name: str
    commit_sha: str


class JustificationOutput(BaseModel):
    evidence: str
    markdown_snippets: str
    sources: list[
        StacktraceSource
        | BreadcrumbsSource
        | HttpRequestSource
        | TraceEventSource
        | ConnectedErrorSource
        | ProfileSource
        | CodeSource
        | DiffSource
    ]
