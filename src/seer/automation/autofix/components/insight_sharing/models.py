from pydantic import BaseModel, Field

from seer.automation.component import BaseComponentOutput


class CodeSnippetContext(BaseModel):
    repo_name: str
    file_path: str
    snippet: str


class BreadcrumbContext(BaseModel):
    type: str
    category: str
    body: str
    level: str
    data_as_json: str


class StacktraceContext(BaseModel):
    file_name: str
    repo_name: str
    function: str
    line_no: int
    col_no: int
    code_snippet: str
    vars_as_json: str


class InsightContextOutput(BaseModel):
    explanation: str
    codebase_context: list[CodeSnippetContext]
    stacktrace_context: list[StacktraceContext]
    event_log_context: list[BreadcrumbContext]


class InsightSharingOutput(BaseComponentOutput):
    insight: str
    codebase_context: list[CodeSnippetContext] = Field(default_factory=list)
    stacktrace_context: list[StacktraceContext] = Field(default_factory=list)
    breadcrumb_context: list[BreadcrumbContext] = Field(default_factory=list)
    justification: str = ""
    generated_at_memory_index: int = -1
