from pydantic import BaseModel, Field

from seer.automation.agent.models import Message
from seer.automation.component import BaseComponentOutput, BaseComponentRequest


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
    error_message_context: list[str]
    codebase_context: list[CodeSnippetContext]
    stacktrace_context: list[StacktraceContext]
    event_log_context: list[BreadcrumbContext]


class InsightSharingRequest(BaseComponentRequest):
    latest_thought: str
    task_description: str
    memory: list[Message]
    past_insights: list[str]
    generated_at_memory_index: int


class InsightSharingOutput(BaseComponentOutput):
    insight: str
    error_message_context: list[str]
    codebase_context: list[CodeSnippetContext]
    stacktrace_context: list[StacktraceContext]
    breadcrumb_context: list[BreadcrumbContext]
    justification: str
    generated_at_memory_index: int = Field(default=-1)
