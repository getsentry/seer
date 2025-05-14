from unittest.mock import MagicMock, patch

import pytest

from seer.automation.autofix.components.insight_sharing.component import process_sources
from seer.automation.autofix.components.insight_sharing.models import (
    BreadcrumbsSource,
    CodeSource,
    ConnectedErrorSource,
    DiffSource,
    HttpRequestSource,
    InsightSources,
    ProfileSource,
    StacktraceSource,
    TraceEventSource,
)
from seer.automation.codebase.repo_client import RepoClient


@pytest.fixture
def mock_context():
    context = MagicMock()
    context.autocorrect_repo_name.side_effect = lambda x: x
    context.autocorrect_file_path.side_effect = lambda path, repo_name: path

    repo_client = MagicMock()
    repo_client.provider = "github"
    repo_client.base_branch = "main"
    repo_client.base_commit_sha = "abcd1234"
    repo_client.repo_full_name = "test-repo"

    repo_client.get_file_url.side_effect = (
        lambda file_path, start_line=None, end_line=None: RepoClient.get_file_url(
            repo_client, file_path, start_line, end_line
        )
    )
    repo_client.get_commit_url.side_effect = lambda commit_sha: RepoClient.get_commit_url(
        repo_client, commit_sha
    )

    context.get_repo_client.return_value = repo_client
    context.get_file_contents.return_value = "sample file content"

    return context


@pytest.fixture
def mock_trace_tree():
    trace_tree = MagicMock()
    trace_tree.trace_id = "test-trace-id"

    def get_event_by_id(event_id):
        if event_id == "trace-event-id":
            event = MagicMock()
            event.event_id = "full-trace-event-id"
            return event
        elif event_id == "error-id":
            event = MagicMock()
            event.event_id = "full-error-id"
            return event
        elif event_id == "profile-event-id":
            event = MagicMock()
            event.event_id = "full-profile-event-id"
            event.profile_id = "profile-123"
            event.project_slug = "test-project"
            return event
        return None

    trace_tree.get_event_by_id.side_effect = get_event_by_id
    return trace_tree


@patch("seer.automation.autofix.components.insight_sharing.component.find_original_snippet")
def test_process_sources(mock_find_snippet, mock_context, mock_trace_tree):
    mock_find_snippet.return_value = ("matched content", 10, 20)

    sources = [
        StacktraceSource(stacktrace_used=True),
        BreadcrumbsSource(breadcrumbs_used=True),
        HttpRequestSource(http_request_used=True),
        TraceEventSource(trace_event_id="trace-event-id"),
        ConnectedErrorSource(connected_error_id="error-id"),
        ProfileSource(trace_event_id="profile-event-id"),
        CodeSource(
            repo_name="test-repo", file_name="test-file.py", code_snippet="def test_func():"
        ),
        DiffSource(repo_name="test-repo", commit_sha="abcd1234"),
    ]

    result = process_sources(sources, mock_context, mock_trace_tree)

    assert isinstance(result, InsightSources)
    assert result.stacktrace_used is True
    assert result.breadcrumbs_used is True
    assert result.http_request_used is True
    assert result.trace_event_ids_used == ["full-trace-event-id"]
    assert result.connected_error_ids_used == ["full-error-id"]
    assert result.profile_ids_used == ["test-project/profile-123"]
    assert len(result.code_used_urls) == 1
    assert (
        result.code_used_urls[0]
        == "https://github.com/test-repo/blob/abcd1234/test-file.py#L10-L20"
    )
    assert len(result.diff_urls) == 1
    assert "https://github.com/test-repo/commit/abcd1234" in result.diff_urls
    assert result.event_trace_id == "test-trace-id"


def test_process_sources_with_empty_sources(mock_context, mock_trace_tree):
    result = process_sources([], mock_context, mock_trace_tree)

    assert isinstance(result, InsightSources)
    assert result.stacktrace_used is False
    assert result.breadcrumbs_used is False
    assert result.http_request_used is False
    assert result.trace_event_ids_used == []
    assert result.connected_error_ids_used == []
    assert result.profile_ids_used == []
    assert result.code_used_urls == []
    assert result.diff_urls == []
    assert result.event_trace_id == "test-trace-id"


def test_process_sources_without_trace_tree(mock_context):
    sources = [
        StacktraceSource(stacktrace_used=True),
        TraceEventSource(trace_event_id="trace-event-id"),  # Should be ignored without trace tree
        CodeSource(repo_name="test-repo", file_name="test-file.py", code_snippet="test"),
    ]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "seer.automation.autofix.utils.find_original_snippet", lambda *args, **kwargs: None
    )

    result = process_sources(sources, mock_context, None)

    assert isinstance(result, InsightSources)
    assert result.stacktrace_used is True
    assert result.trace_event_ids_used == []
    assert len(result.code_used_urls) == 1
    assert "https://github.com/test-repo/blob/abcd1234/test-file.py" in result.code_used_urls
    assert result.event_trace_id is None
