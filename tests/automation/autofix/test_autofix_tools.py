from unittest.mock import ANY, MagicMock, patch

import pytest

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.tools.tools import BaseTools
from seer.automation.models import FileChange, FilePatch, Hunk, Line, RepoDefinition


class TestRepo:
    """A real object to replace complex mocking chains for repositories in tests."""

    def __init__(self, name="test/repo", repo_id="123"):
        self.full_name = name
        self.external_id = repo_id
        self.file_changes = []
        self.codebases = [MagicMock(file_changes=self.file_changes)]

    def add_file_change(self, file_change):
        """Add a file change to this repo."""
        self.file_changes.append(file_change)
        return True


class TestState:
    """A real object to replace complex mocking chains for state in tests."""

    def __init__(self, repos=None):
        self.repos = repos or [TestRepo()]
        self.request = MagicMock()
        self.request.repos = self.repos
        self.readable_repos = self.repos

    def get_repo(self, name):
        """Get a repo by name."""
        for repo in self.repos:
            if repo.full_name == name:
                return repo
        return None


@pytest.fixture
def test_repos():
    """Create test repos for use in tests."""
    return [TestRepo("test/repo", "123")]


@pytest.fixture
def test_state(test_repos):
    """Create a test state with repos for use in tests."""
    return TestState(test_repos)


@pytest.fixture
def autofix_tools(test_state):
    """Create a BaseTools instance with a properly configured context."""
    context = MagicMock(spec=AutofixContext)
    context.event_manager = MagicMock()
    context.state = MagicMock()

    # Use the real test state
    context.state.get.return_value = test_state

    # Set up context methods
    context._get_repo_names = MagicMock(return_value=[repo.full_name for repo in test_state.repos])
    context._attempt_fix_path = MagicMock(return_value="test.py")

    with patch(
        "seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download", MagicMock()
    ):
        tools = BaseTools(context)
    return tools


class TestFileSystem:
    @patch("seer.automation.autofix.tools.tools.cleanup_dir")
    def test_context_manager_cleanup(self, mock_cleanup_dir, autofix_tools: BaseTools):
        with autofix_tools as tools:
            tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert autofix_tools.tmp_dir == {}

    @patch("seer.automation.autofix.tools.tools.cleanup_dir")
    def test_cleanup_method(self, mock_cleanup_dir, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}

        autofix_tools.cleanup()

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert autofix_tools.tmp_dir == {}

    @patch("seer.automation.autofix.tools.tools.cleanup_dir")
    def test_cleanup_not_called_when_tmp_dir_is_none(
        self, mock_cleanup_dir, autofix_tools: BaseTools
    ):
        assert autofix_tools.tmp_dir == {}

        autofix_tools.cleanup()

        mock_cleanup_dir.assert_not_called()
        assert autofix_tools.tmp_dir == {}


class TestSemanticFileSearch:
    @patch("seer.automation.autofix.tools.semantic_search.semantic_search")
    def test_semantic_file_search_found(self, mock_semantic_search, autofix_tools: BaseTools):
        query = "find the main file"
        expected_result = "Found relevant file: `src/main.py`"
        mock_semantic_search.return_value = expected_result

        result = autofix_tools.semantic_file_search(query)

        mock_semantic_search.assert_called_once_with(query=query, context=autofix_tools.context)
        autofix_tools.context.event_manager.add_log.assert_called_with(
            f'Searching for "{query}"...'
        )
        assert result == expected_result

    @patch("seer.automation.autofix.tools.semantic_search.semantic_search")
    def test_semantic_file_search_not_found(self, mock_semantic_search, autofix_tools: BaseTools):
        query = "find nonexistent file"
        expected_result = "Could not figure out which file matches what you were looking for. You'll have to try yourself."
        mock_semantic_search.return_value = ""  # Simulate agent returning nothing

        result = autofix_tools.semantic_file_search(query)

        mock_semantic_search.assert_called_once_with(query=query, context=autofix_tools.context)
        autofix_tools.context.event_manager.add_log.assert_called_with(
            f'Searching for "{query}"...'
        )
        assert result == expected_result


class TestViewDiff:
    def test_view_diff_found(self, autofix_tools: BaseTools):
        autofix_tools.context.__class__ = AutofixContext

        file_path = "src/example.py"
        repo_name = "owner/test_repo"
        commit_sha = "abc1234"
        expected_patch = "@@ -1,5 +1,7 @@\n-old code\n+new code"

        autofix_tools.context.get_commit_patch_for_file.return_value = expected_patch

        result = autofix_tools.view_diff(file_path, repo_name, commit_sha)

        autofix_tools.context.event_manager.add_log.assert_called_with(
            f"Studying commit `{commit_sha}` in `{file_path}` in `{repo_name}`..."
        )
        autofix_tools.context.get_commit_patch_for_file.assert_called_with(
            path=file_path, repo_name=repo_name, commit_sha=commit_sha
        )
        assert result == expected_patch

    def test_view_diff_not_found(self, autofix_tools: BaseTools):
        autofix_tools.context.__class__ = AutofixContext

        file_path = "src/nonexistent.py"
        repo_name = "owner/test_repo"
        commit_sha = "abc1234"

        autofix_tools.context.get_commit_patch_for_file.return_value = None

        result = autofix_tools.view_diff(file_path, repo_name, commit_sha)

        autofix_tools.context.event_manager.add_log.assert_called_with(
            f"Studying commit `{commit_sha}` in `{file_path}` in `{repo_name}`..."
        )
        autofix_tools.context.get_commit_patch_for_file.assert_called_with(
            path=file_path, repo_name=repo_name, commit_sha=commit_sha
        )
        assert (
            result
            == "Could not find the file in the given commit. Either your hash is incorrect or the file does not exist in the given commit."
        )

    def test_view_diff_non_autofix_context(self, autofix_tools: BaseTools):
        autofix_tools.context.__class__ = MagicMock
        result = autofix_tools.view_diff("file.py", "owner/repo", "abc1234")
        assert result is None


class TestExplainFile:
    def test_explain_file_found(self, autofix_tools: BaseTools):
        autofix_tools.context.__class__ = AutofixContext

        file_path = "src/example.py"
        repo_name = "owner/test_repo"
        commit_history = [
            "2023-01-01: abc1234 - Initial commit (Author: Test User)",
            "2023-01-02: def5678 - Update example.py (Author: Another User)",
        ]

        autofix_tools.context.get_commit_history_for_file.return_value = commit_history

        result = autofix_tools.explain_file(file_path, repo_name)

        autofix_tools.context.get_commit_history_for_file.assert_called_with(
            file_path, repo_name, max_commits=30
        )
        expected_result = "COMMIT HISTORY:\n" + "\n".join(commit_history)
        assert result == expected_result

    def test_explain_file_not_found(self, autofix_tools: BaseTools):
        autofix_tools.context.__class__ = AutofixContext

        file_path = "src/nonexistent.py"
        repo_name = "owner/test_repo"

        autofix_tools.context.get_commit_history_for_file.return_value = None

        result = autofix_tools.explain_file(file_path, repo_name)

        autofix_tools.context.get_commit_history_for_file.assert_called_with(
            file_path, repo_name, max_commits=30
        )
        assert (
            result
            == "No commit history found for the given file. Either the file path or repo name is incorrect, or it is just unavailable right now."
        )

    def test_explain_file_empty_history(self, autofix_tools: BaseTools):
        autofix_tools.context.__class__ = AutofixContext

        file_path = "src/new_file.py"
        repo_name = "owner/test_repo"

        autofix_tools.context.get_commit_history_for_file.return_value = []

        result = autofix_tools.explain_file(file_path, repo_name)

        autofix_tools.context.get_commit_history_for_file.assert_called_with(
            file_path, repo_name, max_commits=30
        )
        assert (
            result
            == "No commit history found for the given file. Either the file path or repo name is incorrect, or it is just unavailable right now."
        )

    def test_explain_file_non_autofix_context(self, autofix_tools: BaseTools):
        autofix_tools.context.__class__ = MagicMock
        result = autofix_tools.explain_file("file.py", "owner/repo")
        assert result is None


class TestRipgrep:
    def test_missing_query_returns_error(self, autofix_tools: BaseTools):
        # Mock _ensure_repos_downloaded
        autofix_tools._ensure_repos_downloaded = MagicMock()

        result = autofix_tools.run_ripgrep(query="")
        assert result == "Error: query is required for ripgrep search"
        # Should still attempt download for empty query
        autofix_tools._ensure_repos_downloaded.assert_called_once_with(None)

    def test_single_repo_not_downloaded_returns_error(self, autofix_tools: BaseTools):
        # Setup repo name but don't add to tmp_dir to simulate not downloaded
        repo_name = "test/repo"
        autofix_tools._get_repo_names = MagicMock(return_value=[repo_name])
        autofix_tools._ensure_repos_downloaded = MagicMock()
        autofix_tools.tmp_dir = {}

        result = autofix_tools.run_ripgrep(query="foo", repo_name=repo_name)

        assert result == f"Error: Repository {repo_name} not found or not downloaded"
        autofix_tools._ensure_repos_downloaded.assert_called_once_with(repo_name)

    @patch("seer.automation.autofix.tools.tools.run_ripgrep_in_repo")
    def test_single_repo_success(self, mock_run_ripgrep, autofix_tools: BaseTools):
        # Setup
        repo_name = "test/repo"
        autofix_tools._get_repo_names = MagicMock(return_value=[repo_name])
        autofix_tools._ensure_repos_downloaded = MagicMock()
        autofix_tools.tmp_dir = {repo_name: ("/tmp/foo", "/tmp/foo")}
        mock_run_ripgrep.return_value = "OK"

        # Run test
        result = autofix_tools.run_ripgrep(
            query="bar",
            include_pattern="*.py",
            exclude_pattern="test_*.py",
            case_sensitive=True,
            repo_name=repo_name,
        )

        # Verify ensure_repos_downloaded was called
        autofix_tools._ensure_repos_downloaded.assert_called_once_with(repo_name)

        # Verify ripgrep was called with correct args
        mock_run_ripgrep.assert_called_once()
        actual_repo_dir = mock_run_ripgrep.call_args[0][0]
        actual_cmd = mock_run_ripgrep.call_args[0][1]
        assert actual_repo_dir == "/tmp/foo"
        assert actual_cmd[0] == "rg"
        assert '"bar"' in actual_cmd
        assert "--max-columns" in actual_cmd
        assert "--threads" in actual_cmd
        assert "--ignore-case" not in actual_cmd  # case_sensitive=True
        assert "--glob" in actual_cmd
        assert '"*.py"' in actual_cmd
        assert '"!test_*.py"' in actual_cmd
        assert actual_cmd[-1] == "/tmp/foo"
        assert result == "OK"

    @patch("seer.automation.autofix.tools.tools.run_ripgrep_in_repo")
    def test_ignore_case_flag_when_not_case_sensitive(
        self, mock_run_ripgrep, autofix_tools: BaseTools
    ):
        # Setup
        repo_name = "test/repo"
        autofix_tools._get_repo_names = MagicMock(return_value=[repo_name])
        autofix_tools._ensure_repos_downloaded = MagicMock()
        autofix_tools.tmp_dir = {repo_name: ("/tmp", "/tmp")}
        mock_run_ripgrep.return_value = ""

        # Run test with default case_sensitive=False
        autofix_tools.run_ripgrep(query="q", repo_name=repo_name)

        # Verify --ignore-case flag was included
        actual_cmd = mock_run_ripgrep.call_args[0][1]
        assert "--ignore-case" in actual_cmd

    @patch("seer.automation.autofix.tools.tools.run_ripgrep_in_repo")
    def test_multiple_repos_combines_results(self, mock_run_ripgrep, autofix_tools: BaseTools):
        # Setup two repos
        repos = ["owner/repo1", "owner/repo2"]
        autofix_tools._get_repo_names = MagicMock(return_value=repos)
        autofix_tools._ensure_repos_downloaded = MagicMock()
        autofix_tools.tmp_dir = {"owner/repo1": ("x", "/d1"), "owner/repo2": ("y", "/d2")}

        # Mock different results for each repo
        def fake_run(repo_dir, cmd):
            if repo_dir == "/d1":
                return "A"
            elif repo_dir == "/d2":
                return "B"
            return None

        mock_run_ripgrep.side_effect = fake_run

        # Run test
        result = autofix_tools.run_ripgrep(query="xyz")

        # Verify single download call for all repos
        autofix_tools._ensure_repos_downloaded.assert_called_once_with(None)

        # Verify results were combined correctly
        expected = "Result for owner/repo1:\nA\nResult for owner/repo2:\nB"
        assert result == expected

    @patch("seer.automation.autofix.tools.tools.run_ripgrep_in_repo")
    def test_no_results_found(self, mock_run_ripgrep, autofix_tools: BaseTools):
        # Setup
        repo_name = "test/repo"
        autofix_tools._get_repo_names = MagicMock(return_value=[repo_name])
        autofix_tools._ensure_repos_downloaded = MagicMock()
        autofix_tools.tmp_dir = {repo_name: ("/tmp/foo", "/tmp/foo")}
        mock_run_ripgrep.return_value = "ripgrep returned: No results found."

        # Run test
        result = autofix_tools.run_ripgrep(query="nonexistent")

        # Verify ripgrep was called
        mock_run_ripgrep.assert_called_once()
        # Verify empty result is handled correctly
        assert result == f"Result for {repo_name}:\nripgrep returned: No results found."


class TestFindFiles:
    def test_find_files_validation(self, autofix_tools: BaseTools):
        result = autofix_tools.find_files("invalid command")
        assert result == "Command must be a valid find command that starts with 'find'."

    @patch("subprocess.run")
    def test_find_files_success(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "./src/file1.py\n./src/dir/file2.py"
        mock_run.return_value = mock_process

        result = autofix_tools.find_files("find . -name '*.py'")

        mock_run.assert_called_once()
        assert mock_run.call_args[1]["shell"] is False
        assert mock_run.call_args[1]["cwd"] == "/tmp/test_dir/repo"
        assert "Results from owner/test_repo:" in result
        assert "./src/file1.py" in result
        assert "./src/dir/file2.py" in result

    @patch("subprocess.run")
    def test_find_files_no_results(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        result = autofix_tools.find_files("find . -name '*.nonexistent'")

        assert "Results from owner/test_repo: no files found." in result

    @patch("subprocess.run")
    def test_find_files_error(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "find: invalid option -- z"
        mock_run.return_value = mock_process

        result = autofix_tools.find_files("find . -z '*.py'")

        assert "Results from owner/test_repo: find: invalid option -- z" in result

    @patch("subprocess.run")
    def test_find_files_multiple_repos(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {
            "owner/repo1": ("/tmp/test_dir1", "/tmp/test_dir1/repo"),
            "owner/repo2": ("/tmp/test_dir2", "/tmp/test_dir2/repo"),
        }
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/repo1", "owner/repo2"])

        def side_effect(cmd, **kwargs):
            cwd = kwargs.get("cwd")
            mock_process = MagicMock()

            if cwd == "/tmp/test_dir1/repo":
                mock_process.returncode = 0
                mock_process.stdout = "./repo1_file.py"
            else:  # repo2
                mock_process.returncode = 0
                mock_process.stdout = "./repo2_file.py"

            return mock_process

        mock_run.side_effect = side_effect

        result = autofix_tools.find_files("find . -name '*.py'")

        assert "Results from owner/repo1" in result
        assert "./repo1_file.py" in result
        assert "Results from owner/repo2" in result
        assert "./repo2_file.py" in result

    @patch("seer.automation.autofix.tools.tools.BaseTools._ensure_repos_downloaded")
    def test_find_files_specific_repo(self, mock_ensure_repos, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}

        with patch("subprocess.run") as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "./file.py"
            mock_run.return_value = mock_process

            autofix_tools.find_files("find . -name '*.py'", repo_name="owner/test_repo")

            mock_ensure_repos.assert_called_once_with("owner/test_repo")
            mock_run.assert_called_once()


class TestParallelRepoDownload:
    @patch("seer.automation.autofix.tools.tools.copy_modules_initializer")
    @patch("seer.automation.autofix.tools.tools.ThreadPoolExecutor")
    def test_init_starts_parallel_download(self, mock_executor_class, mock_copy_initializer):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor
        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = [MagicMock(full_name="owner/repo1")]

        # Create instance without patching _start_parallel_repo_download to test it
        tools = BaseTools(context)

        # Verify the executor was created with the correct initializer
        mock_executor_class.assert_called_once_with(initializer=mock_copy_initializer.return_value)

        # Verify the parallel download was started
        assert mock_executor.submit.called
        assert tools._download_future is not None

    @patch("seer.automation.autofix.tools.tools.ThreadPoolExecutor")
    def test_cleanup_cancels_download_future(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor
        mock_future = MagicMock()

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = [MagicMock(full_name="owner/repo1")]

        # Patch _start_parallel_repo_download to avoid starting real threads
        with patch("seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Set a mock future
        tools._download_future = mock_future
        tools._download_future.done.return_value = False

        # Call cleanup
        tools.cleanup()

        # Verify future was cancelled
        mock_future.cancel.assert_called_once()
        assert tools._download_future is None

    @patch("seer.automation.autofix.tools.tools.ThreadPoolExecutor")
    def test_exit_shuts_down_executor(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = []

        # Patch _start_parallel_repo_download to avoid starting real threads
        with patch("seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Call __exit__
        tools.__exit__(None, None, None)

        # Verify executor was shut down properly
        mock_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_ensure_repos_downloaded_with_completed_future(self):
        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = [MagicMock(full_name="owner/repo1")]

        # Patch _start_parallel_repo_download to avoid starting real threads
        with patch("seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Setup a completed future
        mock_future = MagicMock()
        mock_future.done.return_value = True
        tools._download_future = mock_future

        # Setup _get_repo_names to return a repo that's not yet in tmp_dir
        tools._get_repo_names = MagicMock(return_value=["owner/repo1"])
        tools.tmp_dir = {}

        # Mock the repo client and its load_repo_to_tmp_dir method
        mock_repo_client = MagicMock()
        mock_repo_client.load_repo_to_tmp_dir.return_value = ("/tmp/dir", "/tmp/dir/repo")
        context.get_repo_client.return_value = mock_repo_client

        # Call the method, but make sure to handle the case where future gets set to None
        # during method execution (which is expected behavior)
        with patch.object(tools, "_download_future", mock_future):
            tools._ensure_repos_downloaded()

        mock_future.result.assert_called_once()
        context.get_repo_client.assert_called_with(
            repo_name="owner/repo1", type=tools.repo_client_type
        )
        mock_repo_client.load_repo_to_tmp_dir.assert_called_once()

        # Verify tmp_dir was updated
        assert "owner/repo1" in tools.tmp_dir
        assert tools.tmp_dir["owner/repo1"] == ("/tmp/dir", "/tmp/dir/repo")

    @patch("seer.automation.autofix.tools.tools.ThreadPoolExecutor")
    @patch("seer.automation.autofix.tools.tools.as_completed")
    def test_ensure_repos_downloaded_parallel(self, mock_as_completed, mock_executor_class):
        # Setup ThreadPoolExecutor context manager mock
        thread_pool_cm = MagicMock()
        thread_pool_executor = MagicMock()
        thread_pool_cm.__enter__.return_value = thread_pool_executor
        mock_executor_class.return_value = thread_pool_cm

        # Create context with multiple repos
        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = [
            MagicMock(full_name="owner/repo1"),
            MagicMock(full_name="owner/repo2"),
            MagicMock(full_name="owner/repo3"),
        ]

        # We need to patch out the _download_future because our test focuses on _ensure_repos_downloaded
        # Create tools with a pre-completed download_future to avoid extra ThreadPoolExecutor usage
        with patch("seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Setup a completed future to avoid triggering more ThreadPoolExecutor creation
        tools._download_future = None

        # Set up repos to download (none in tmp_dir yet)
        tools.tmp_dir = {}
        tools._get_repo_names = MagicMock(
            return_value=["owner/repo1", "owner/repo2", "owner/repo3"]
        )

        # Mock repo client and futures
        mock_repo_client = MagicMock()
        mock_repo_client.load_repo_to_tmp_dir.return_value = ("/tmp/dir", "/tmp/dir/repo")
        context.get_repo_client.return_value = mock_repo_client

        # Setup the Future results
        future1 = MagicMock()
        future1.result.return_value = ("owner/repo1", ("/tmp/dir1", "/tmp/dir1/repo"))

        future2 = MagicMock()
        future2.result.return_value = ("owner/repo2", ("/tmp/dir2", "/tmp/dir2/repo"))

        future3 = MagicMock()
        future3.result.return_value = ("owner/repo3", ("/tmp/dir3", "/tmp/dir3/repo"))

        # Mock thread_pool_executor.submit to return our futures
        thread_pool_executor.submit.side_effect = [future1, future2, future3]

        # Mock as_completed to yield our futures in order
        mock_as_completed.return_value = [future1, future2, future3]

        # Reset the mock to clear any previous calls
        mock_executor_class.reset_mock()

        # Call method
        with patch("seer.automation.autofix.tools.tools.append_langfuse_observation_metadata"):
            tools._ensure_repos_downloaded()

        # Verify ThreadPoolExecutor was created exactly once during the test
        mock_executor_class.assert_called_once()

        # Verify submit was called three times (once for each repo)
        assert thread_pool_executor.submit.call_count == 3

        # Verify all repos were added to tmp_dir
        assert len(tools.tmp_dir) == 3
        assert "owner/repo1" in tools.tmp_dir
        assert "owner/repo2" in tools.tmp_dir
        assert "owner/repo3" in tools.tmp_dir

    @patch("seer.automation.autofix.tools.tools.ThreadPoolExecutor")
    def test_cleanup_handles_future_cancel_exception(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = []

        # Create tools
        with patch("seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Set up a future that raises an exception when cancelled
        mock_future = MagicMock()
        mock_future.done.return_value = False
        mock_future.cancel.side_effect = Exception("Cancel failed")
        tools._download_future = mock_future

        # Call cleanup - should handle the exception gracefully
        with patch("seer.automation.autofix.tools.tools.logger.exception") as mock_logger:
            tools.cleanup()

        # Verify exception was logged
        mock_logger.assert_called_once()
        assert "Cancel failed" in str(mock_logger.call_args)
        assert tools._download_future is None

    @patch("seer.automation.autofix.tools.tools.ThreadPoolExecutor")
    def test_exit_handles_executor_shutdown_exception(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor
        mock_executor.shutdown.side_effect = Exception("Shutdown failed")

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = []

        # Create tools
        with patch("seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Call __exit__ - should handle the exception gracefully
        with patch("seer.automation.autofix.tools.tools.logger.exception") as mock_logger:
            tools.__exit__(None, None, None)

        # Verify exception was logged
        mock_logger.assert_called_once()
        assert "Shutdown failed" in str(mock_logger.call_args)
        mock_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    @patch("seer.automation.autofix.tools.tools.ThreadPoolExecutor")
    def test_ensure_repos_downloaded_skips_already_downloaded(self, mock_executor_class):
        # Setup mocks
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor

        # Create context
        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = [
            MagicMock(full_name="owner/repo1"),
            MagicMock(full_name="owner/repo2"),
        ]

        # Create tools with some repos already in tmp_dir
        with patch("seer.automation.autofix.tools.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Set up tmp_dir to simulate already downloaded repo
        tools.tmp_dir = {"owner/repo1": ("/tmp/dir1", "/tmp/dir1/repo")}
        tools._get_repo_names = MagicMock(return_value=["owner/repo1", "owner/repo2"])

        # Mock repo client
        mock_repo_client = MagicMock()
        mock_repo_client.load_repo_to_tmp_dir.return_value = ("/tmp/dir2", "/tmp/dir2/repo")
        context.get_repo_client.return_value = mock_repo_client

        # Call method
        with patch("seer.automation.autofix.tools.tools.append_langfuse_observation_metadata"):
            tools._ensure_repos_downloaded()

        # Verify only the second repo was downloaded
        context.get_repo_client.assert_called_once_with(
            repo_name="owner/repo2", type=tools.repo_client_type
        )
        mock_repo_client.load_repo_to_tmp_dir.assert_called_once()
        assert "owner/repo1" in tools.tmp_dir
        assert "owner/repo2" in tools.tmp_dir


class TestClaudeTools:
    def test_handle_view_command(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools.context.get_file_contents.return_value = "line1\nline2\nline3"
        kwargs = {"view_range": ["1", "2"]}

        # Test
        result = autofix_tools._handle_view_command(kwargs, "test/repo", "test.py")

        # Assert
        assert "1: line1" in result
        assert "2: line2" in result
        assert "line3" not in result

    def test_handle_view_command_invalid_range(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools.context.get_file_contents.return_value = "line1\nline2\nline3"
        kwargs = {"view_range": ["2", "1"]}  # Invalid range

        # Test
        result = autofix_tools._handle_view_command(kwargs, "test/repo", "test.py")

        # Assert
        assert "Invalid line range" in result

    def test_handle_view_command_file_not_found(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools.context.get_file_contents.return_value = None
        kwargs = {}

        # Test
        result = autofix_tools._handle_view_command(kwargs, "test/repo", "test.py")

        # Assert
        assert "File not found" in result

    def test_handle_str_replace_command(self, autofix_tools: BaseTools, test_state, test_repos):
        # Setup
        autofix_tools.context.get_file_contents.return_value = "old text"
        kwargs = {"old_str": "old text", "new_str": "new text"}

        # Set up the actual method under test to use our mocked state
        autofix_tools.context.state.update.return_value.__enter__.return_value = test_state

        # Since _append_file_change is a method of BaseTools, we mock it to avoid modifying the real state
        autofix_tools._append_file_change = MagicMock(return_value=True)

        # Test
        result = autofix_tools._handle_str_replace_command(kwargs, "test/repo", "test.py")

        # Assert
        assert "Change applied successfully" in result
        # Verify we called _append_file_change with the right repo name
        autofix_tools._append_file_change.assert_called_once_with("test/repo", ANY)
        # Verify the file change was correctly constructed
        actual_file_change = autofix_tools._append_file_change.call_args[0][1]
        assert actual_file_change.change_type == "edit"
        assert actual_file_change.reference_snippet == "old text"
        assert actual_file_change.new_snippet == "new text"

    def test_handle_create_command(self, autofix_tools: BaseTools):
        # Setup
        kwargs = {"file_text": "new file content"}
        autofix_tools.context.get_file_contents.return_value = None

        # Setup proper request.repos structure
        mock_repo = MagicMock(full_name="test/repo", external_id="123")

        # Make file_changes a regular list instead of a MagicMock
        file_changes = []
        mock_codebase = MagicMock()
        mock_codebase.file_changes = file_changes
        mock_repo.codebases = [mock_codebase]

        mock_state = MagicMock()
        mock_state.request.repos = [mock_repo]

        # Create a custom side effect to actually append to the file_changes list
        def append_file_change(*args, **kwargs):
            for repo in mock_state.request.repos:
                if repo.full_name == "test/repo":
                    repo.codebases[0].file_changes.append(
                        FileChange(
                            change_type="create",
                            path="test.py",
                            reference_snippet="new file content",
                            new_snippet="new file content",
                            repo_name="test/repo",
                        )
                    )
                    return True
            return False

        # Mock _append_file_change with our custom implementation
        autofix_tools._append_file_change = MagicMock(side_effect=append_file_change)
        autofix_tools.context.state.update.return_value.__enter__.return_value = mock_state

        # Test
        result = autofix_tools._handle_create_command(kwargs, "test/repo", "test.py")

        # Assert
        assert "Change applied successfully" in result
        assert len(mock_repo.codebases[0].file_changes) == 1

    def test_handle_create_command_missing_text(self, autofix_tools: BaseTools):
        # Test
        result = autofix_tools._handle_create_command({}, "test/repo", "test.py")

        # Assert
        assert "file_text is required" in result

    def test_handle_insert_command(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools.context.get_file_contents.return_value = "line1\nline2"
        kwargs = {"insert_line": "1", "insert_text": "new line"}

        # Setup proper request.repos structure
        mock_repo = MagicMock(full_name="test/repo", external_id="123")

        # Make file_changes a regular list instead of a MagicMock
        file_changes = []
        mock_codebase = MagicMock()
        mock_codebase.file_changes = file_changes
        mock_repo.codebases = [mock_codebase]

        mock_state = MagicMock()
        mock_state.request.repos = [mock_repo]

        # Create a custom side effect to actually append to the file_changes list
        def append_file_change(*args, **kwargs):
            for repo in mock_state.request.repos:
                if repo.full_name == "test/repo":
                    repo.codebases[0].file_changes.append(
                        FileChange(
                            change_type="edit",
                            path="test.py",
                            reference_snippet="line1\nline2",
                            new_snippet="line1\nnew line\nline2",
                            repo_name="test/repo",
                        )
                    )
                    return True
            return False

        # Mock _append_file_change with our custom implementation
        autofix_tools._append_file_change = MagicMock(side_effect=append_file_change)
        autofix_tools.context.state.update.return_value.__enter__.return_value = mock_state

        # Test
        result = autofix_tools._handle_insert_command(kwargs, "test/repo", "test.py")

        # Assert
        assert "Change applied successfully" in result
        assert len(mock_repo.codebases[0].file_changes) == 1

    def test_handle_insert_command_invalid_line(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools.context.get_file_contents.return_value = "line1\nline2"
        kwargs = {"insert_line": "5", "insert_text": "new line"}

        # Test
        result = autofix_tools._handle_insert_command(kwargs, "test/repo", "test.py")

        # Assert
        assert "Invalid line number" in result

    def test_handle_insert_command_missing_params(self, autofix_tools: BaseTools):
        # Test
        result = autofix_tools._handle_insert_command({}, "test/repo", "test.py")

        # Assert
        assert "insert_line is required" in result

    def test_handle_undo_edit_command(self, autofix_tools: BaseTools):
        # Setup
        file_change = FileChange(
            change_type="edit",
            commit_message="test",
            reference_snippet="old",
            new_snippet="new",
            path="test.py",
            repo_name="test/repo",
        )

        # Setup proper request.repos structure
        mock_repo = MagicMock(full_name="test/repo", external_id="123")
        mock_codebase = MagicMock(file_changes=[file_change])

        mock_state = MagicMock()
        mock_state.codebases = {
            "123": mock_codebase,
        }
        mock_state.request.repos = [mock_repo]

        autofix_tools.context.state.update.return_value.__enter__.return_value = mock_state

        # Test
        result = autofix_tools._handle_undo_edit_command({}, "test/repo", "test.py")

        # Assert
        assert "File changes undone successfully" in result
        assert len(mock_state.request.repos[0].codebases[0].file_changes) == 0

    def test_handle_undo_edit_command_no_changes(self, autofix_tools: BaseTools):
        # Setup
        mock_repo = MagicMock(full_name="test/repo", external_id="123")
        mock_codebase = MagicMock(file_changes=[])
        mock_repo.codebases = [mock_codebase]

        mock_state = MagicMock()
        mock_state.request.repos = []  # Set empty repos to trigger "No file changes found to undo"

        autofix_tools.context.state.update.return_value.__enter__.return_value = mock_state

        # Test
        result = autofix_tools._handle_undo_edit_command({}, "test/repo", "test.py")

        # Assert
        assert "No file changes found to undo" in result

    def test_handle_unknown_command(self, autofix_tools: BaseTools):
        # Directly mock the _get_repo_name_and_path method to return success
        autofix_tools._get_repo_name_and_path = MagicMock(
            return_value=(None, "test/repo", "test.py")
        )

        # Test
        result = autofix_tools.handle_claude_tools(command="unknown", path="test.py")

        # Assert
        assert "Unknown command" in result

    def test_handle_claude_tools_with_error(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools.context._get_repo_names = MagicMock(return_value=["test/repo"])
        autofix_tools.context._attempt_fix_path = MagicMock(return_value=None)

        # Test
        result = autofix_tools.handle_claude_tools(command="view", path="invalid/path")

        # Assert
        assert "does not exist in the repository" in result

    def test_handle_claude_tools_success(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools._get_repo_name_and_path = MagicMock(
            return_value=(None, "test/repo", "test.py")
        )
        autofix_tools.context.get_file_contents.return_value = "test content"

        # This function overrides to directly return the view result
        def handle_view_command(kwargs, repo_name, path, tool_call_id, current_memory_index):
            return "test content"

        # Mock the _handle_view_command to avoid calling the real implementation
        autofix_tools._handle_view_command = MagicMock(side_effect=handle_view_command)

        # Test
        result = autofix_tools.handle_claude_tools(command="view", path="test.py")

        # Assert
        assert "test content" in result

    def test_handle_claude_tools_multiple_repos(self, autofix_tools: BaseTools):
        # Setup - directly mock the _get_repo_name_and_path method
        autofix_tools._get_repo_name_and_path = MagicMock(
            return_value=(None, "test/repo1", "test.py")
        )
        autofix_tools.context.get_file_contents.return_value = "test content"

        # Mock the view command handler
        def handle_view_command(kwargs, repo_name, path, tool_call_id, current_memory_index):
            return "test content"

        autofix_tools._handle_view_command = MagicMock(side_effect=handle_view_command)

        # Test
        result = autofix_tools.handle_claude_tools(command="view", path="test/repo1:test.py")

        # Assert
        assert "test content" in result

    def test_handle_claude_tools_multiple_repos_no_repo_specified(self, autofix_tools: BaseTools):
        # Setup
        autofix_tools.context._get_repo_names = MagicMock(return_value=["test/repo1", "test/repo2"])

        # Clear the previous mock of _attempt_fix_path if any
        if hasattr(autofix_tools.context._attempt_fix_path, "reset_mock"):
            autofix_tools.context._attempt_fix_path.reset_mock()

        # Don't mock _attempt_fix_path since we want the validation to fail on multiple repos
        # Instead, pass the real error from _get_repo_name_and_path

        # Original implementation of _get_repo_name_and_path
        original_get_repo_name_and_path = autofix_tools._get_repo_name_and_path

        # Mock to ensure the multiple repos error is returned
        def mock_get_repo_name_and_path(kwargs, allow_nonexistent_paths=False):
            # Extract the path argument safely
            path_arg = kwargs.get("path", "")
            # Check if multiple repos and path doesn't specify one
            if len(autofix_tools.context._get_repo_names()) > 1 and ":" not in path_arg:
                return (
                    "Error: Multiple repositories found. Please provide a repository name in the format "
                    "`repo_name:path`, such as `repo_owner/repo:src/foo/bar.py`. The repositories available "
                    f"to you are: {', '.join(autofix_tools.context._get_repo_names())}",
                    None,
                    None,
                )
            return original_get_repo_name_and_path(kwargs)

        autofix_tools._get_repo_name_and_path = MagicMock(side_effect=mock_get_repo_name_and_path)

        # Test
        result = autofix_tools.handle_claude_tools(command="view", path="test.py")

        # Assert
        assert "Multiple repositories found" in result

    def test_handle_str_replace_command_missing_params(self, autofix_tools: BaseTools):
        # Test
        result = autofix_tools._handle_str_replace_command({}, "test/repo", "test.py")

        # Assert
        assert "old_str and new_str are required" in result

    def test_handle_claude_tools_create_nonexistent_path(self, autofix_tools: BaseTools):
        """Verify that the 'create' command works even if the path doesn't exist yet."""
        # Setup
        repo_name = "test/repo"
        new_path = "new/path/to/create.py"
        file_text = "This is the new file content."
        kwargs = {"command": "create", "path": new_path, "file_text": file_text}

        # Mock _get_repo_names to return a single repo
        autofix_tools.context._get_repo_names = MagicMock(return_value=[repo_name])
        # Mock _attempt_fix_path to return None, simulating the path not existing
        # Note: This mock is on the context, as that's where _get_repo_name_and_path calls it
        autofix_tools._attempt_fix_path = MagicMock(return_value=None)
        # Mock get_file_contents to return None (as the file shouldn't exist yet)
        autofix_tools.context.get_file_contents = MagicMock(return_value=None)
        # Mock _append_file_change to simulate successful application
        autofix_tools._append_file_change = MagicMock(return_value=True)
        # Mock make_file_patches to return a proper FilePatch for file creation
        with patch("seer.automation.autofix.tools.tools.make_file_patches") as mock_make_patches:
            # Create a proper FilePatch for file creation
            file_patch = FilePatch(
                type="A",
                path=new_path,
                added=1,
                removed=0,
                source_file="/dev/null",
                target_file=new_path,
                hunks=[
                    Hunk(
                        source_start=0,
                        source_length=0,
                        target_start=1,
                        target_length=1,
                        section_header="@@ -0,0 +1,1 @@",
                        lines=[
                            Line(target_line_no=1, value=file_text, line_type="+"),
                        ],
                    )
                ],
            )
            mock_make_patches.return_value = ([file_patch], "")

            # Mock event manager's send_insight method
            autofix_tools.context.event_manager.send_insight = MagicMock()
            autofix_tools.context.event_manager.add_log = MagicMock()

            # Test
            result = autofix_tools.handle_claude_tools(**kwargs)

            # Assert
            # 1. Check that _attempt_fix_path was called for the new path (it should be)
            autofix_tools._attempt_fix_path.assert_called_once_with(new_path, repo_name)
            # 2. Check that get_file_contents was called (by _handle_create_command)
            autofix_tools.context.get_file_contents.assert_called_once_with(
                new_path, repo_name=repo_name
            )
            # 3. Check that make_file_patches was called
            mock_make_patches.assert_called_once()
            # 4. Check that _append_file_change was called
            autofix_tools._append_file_change.assert_called_once()
            # 5. Check that the result indicates success (not a path error)
            assert "Change applied successfully" in result
            # 6. Check event manager logs and insights were called
            autofix_tools.context.event_manager.add_log.assert_called()
            autofix_tools.context.event_manager.send_insight.assert_called_once()

    def test_handle_claude_tools_view_nonexistent_path_fails(self, autofix_tools: BaseTools):
        """Verify that commands other than 'create'/'undo_edit' still fail for non-existent paths."""
        # Setup
        repo_name = "test/repo"
        nonexistent_path = "non/existent/path.py"
        kwargs = {"command": "view", "path": nonexistent_path}  # Use 'view' command

        # Mock _get_repo_names to return a single repo
        autofix_tools.context._get_repo_names = MagicMock(return_value=[repo_name])
        # Mock _attempt_fix_path to return None, simulating the path not existing
        autofix_tools._attempt_fix_path = MagicMock(return_value=None)

        # Test
        result = autofix_tools.handle_claude_tools(**kwargs)

        # Assert
        # Check that _attempt_fix_path was called
        autofix_tools._attempt_fix_path.assert_called_once_with(nonexistent_path, repo_name)
        # Check that the result is the expected path error message
        assert (
            f"Error: The path you provided '{nonexistent_path}' does not exist in the repository '{repo_name}'."
            in result
        )
        # Ensure the view handler wasn't called (it shouldn't be if the path check fails)
        autofix_tools.context.get_file_contents.assert_not_called()


class TestViewDirectoryTree:
    def test_view_directory_tree_empty(self, autofix_tools: BaseTools):
        """Test viewing an empty directory tree"""
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])
        autofix_tools.repo_client = MagicMock()
        autofix_tools.repo_client.get_valid_file_paths.return_value = set()

        result = autofix_tools.tree("src")
        assert "<no entries found in directory 'src'/>" in result

    def test_view_directory_tree_small(self, autofix_tools: BaseTools):
        """Test viewing a small directory tree that doesn't exceed the file limit"""
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_repo_client = MagicMock()
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        # Create a small set of files (less than MAX_FILES_IN_TREE)
        files = {
            "src/main.py",
            "src/utils/helper.py",
            "src/utils/__init__.py",
            "tests/test_main.py",
        }
        mock_repo_client.get_valid_file_paths.return_value = files
        mock_repo_client._build_file_tree_string.return_value = "mock tree string"

        result = autofix_tools.tree("src")
        assert "<directory_tree>" in result
        assert "mock tree string" in result
        assert "Notice: There are a total of" not in result

    def test_view_directory_tree_large(self, autofix_tools: BaseTools):
        """Test viewing a large directory tree that exceeds the file limit"""
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_repo_client = MagicMock()
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        # Create a large set of files (more than MAX_FILES_IN_TREE)
        files = {f"src/file{i}.py" for i in range(150)}  # 150 files > MAX_FILES_IN_TREE (100)
        mock_repo_client.get_valid_file_paths.return_value = files
        mock_repo_client._build_file_tree_string.return_value = "mock tree string"

        result = autofix_tools.tree("src")
        assert "<directory_tree>" in result
        assert "mock tree string" in result
        assert "Notice: There are a total of 150 files in the tree" in result
        assert "provide a more specific path to view a full tree" in result


class TestExpandDocument:
    def test_expand_document_fallback_to_read_file_contents(self, autofix_tools: BaseTools):
        """Test that expand_document falls back to read_file_contents when get_file_contents returns None"""
        # Setup
        repo_name = "test/repo"
        file_path = "src/test.py"
        expected_content = "test file content"

        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = repo_name

        # Mock attempt_fix_path to return the file path
        autofix_tools._attempt_fix_path = MagicMock(return_value=file_path)

        # Mock get_file_contents to return None to trigger fallback
        autofix_tools.context.get_file_contents = MagicMock(return_value=None)

        # Mock _ensure_repos_downloaded
        autofix_tools._ensure_repos_downloaded = MagicMock()

        # Setup tmp_dir to simulate downloaded repo
        autofix_tools.tmp_dir = {repo_name: ("/tmp/test", "/tmp/test/repo")}

        # Mock read_file_contents to return content
        with patch("seer.automation.autofix.tools.tools.read_file_contents") as mock_read:
            mock_read.return_value = (expected_content, None)

            # Call expand_document
            result = autofix_tools.expand_document(file_path, repo_name)

            # Verify fallback behavior
            autofix_tools.context.get_file_contents.assert_called_once_with(
                file_path, repo_name=repo_name
            )
            autofix_tools._ensure_repos_downloaded.assert_called_once_with(repo_name)
            mock_read.assert_called_once_with("/tmp/test/repo", file_path)
            assert result == expected_content

    def test_expand_document_fallback_with_read_error(self, autofix_tools: BaseTools):
        """Test that expand_document handles errors from read_file_contents properly"""
        # Setup
        repo_name = "test/repo"
        file_path = "src/test.py"
        error_msg = "File does not exist"

        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = repo_name

        # Mock attempt_fix_path to return the file path
        autofix_tools._attempt_fix_path = MagicMock(return_value=file_path)

        # Mock get_file_contents to return None to trigger fallback
        autofix_tools.context.get_file_contents = MagicMock(return_value=None)

        # Mock _ensure_repos_downloaded
        autofix_tools._ensure_repos_downloaded = MagicMock()

        # Setup tmp_dir to simulate downloaded repo
        autofix_tools.tmp_dir = {repo_name: ("/tmp/test", "/tmp/test/repo")}

        # Mock read_file_contents to return error
        with patch("seer.automation.autofix.tools.tools.read_file_contents") as mock_read:
            mock_read.return_value = (None, error_msg)

            # Call expand_document
            result = autofix_tools.expand_document(file_path, repo_name)

            # Verify error handling
            autofix_tools.context.get_file_contents.assert_called_once_with(
                file_path, repo_name=repo_name
            )
            autofix_tools._ensure_repos_downloaded.assert_called_once_with(repo_name)
            mock_read.assert_called_once_with("/tmp/test/repo", file_path)
            assert "Error: Could not read the file at path" in result
            assert error_msg in result

    def test_expand_document_fallback_repo_not_downloaded(self, autofix_tools: BaseTools):
        """Test that expand_document handles case where repo is not in tmp_dir"""
        # Setup
        repo_name = "test/repo"
        file_path = "src/test.py"

        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = repo_name

        # Mock attempt_fix_path to return the file path
        autofix_tools._attempt_fix_path = MagicMock(return_value=file_path)

        # Mock get_file_contents to return None to trigger fallback
        autofix_tools.context.get_file_contents = MagicMock(return_value=None)

        # Mock _ensure_repos_downloaded
        autofix_tools._ensure_repos_downloaded = MagicMock()

        # Empty tmp_dir to simulate repo not downloaded
        autofix_tools.tmp_dir = {}

        # Mock read_file_contents (should not be called)
        with patch("seer.automation.autofix.tools.tools.read_file_contents") as mock_read:
            # Call expand_document
            result = autofix_tools.expand_document(file_path, repo_name)

            # Verify behavior
            autofix_tools.context.get_file_contents.assert_called_once_with(
                file_path, repo_name=repo_name
            )
            autofix_tools._ensure_repos_downloaded.assert_called_once_with(repo_name)
            mock_read.assert_not_called()
            assert "Error: Could not read the file at path" in result

    def test_expand_document_success_without_fallback(self, autofix_tools: BaseTools):
        """Test that expand_document returns content from get_file_contents when available"""
        # Setup
        repo_name = "test/repo"
        file_path = "src/test.py"
        expected_content = "test file content"

        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = repo_name

        # Mock attempt_fix_path to return the file path
        autofix_tools._attempt_fix_path = MagicMock(return_value=file_path)

        # Mock get_file_contents to return content
        autofix_tools.context.get_file_contents = MagicMock(return_value=expected_content)

        # Mock _ensure_repos_downloaded
        autofix_tools._ensure_repos_downloaded = MagicMock()

        # Mock read_file_contents (should not be called)
        with patch("seer.automation.autofix.tools.tools.read_file_contents") as mock_read:
            # Call expand_document
            result = autofix_tools.expand_document(file_path, repo_name)

            # Verify no fallback was needed
            autofix_tools.context.get_file_contents.assert_called_once_with(
                file_path, repo_name=repo_name
            )
            autofix_tools._ensure_repos_downloaded.assert_not_called()
            mock_read.assert_not_called()
            assert result == expected_content

    def test_incorrect_repo_name_returns_error(self, autofix_tools: BaseTools):
        """Test that expand_document returns an error when the repo name is incorrect"""
        # Setup
        repo_name = "test/repo"
        file_path = "src/test.py"
        error_msg = "File does not exist"

        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = None

        autofix_tools.context.repos = [
            RepoDefinition(
                provider="github",
                owner="valid",
                name="repo",
                external_id="123",
            )
        ]

        result = autofix_tools.expand_document(file_path, repo_name)

        # Assert
        assert "Error: Repo 'test/repo' not found" in result
        assert "valid/repo" in result
