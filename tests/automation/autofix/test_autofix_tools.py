from concurrent.futures import Future
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.models import AutofixRequest
from seer.automation.autofix.tools.tools import FIRST_EXCEPTION, BaseTools, RepoClientType
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
    context.autocorrect_repo_name = MagicMock(return_value="test/repo")

    with patch("seer.automation.autofix.tools.tools.BaseTools._download_repos", MagicMock()):
        tools = BaseTools(context)
    return tools


class TestDownloadRepos:
    @patch("seer.automation.autofix.tools.tools.RepoManager")
    def test_download_repos_basic(self, mock_repo_manager):
        """Test that _download_repos creates and initializes repo managers correctly."""
        # Setup
        context = MagicMock()
        repo_names = ["owner/repo1", "owner/repo2"]
        repo_clients = {name: MagicMock() for name in repo_names}

        # Mock the _get_repo_names method directly to avoid state.get() issues
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=repo_names)
        tools.retrieval_top_k = 8
        tools.repo_client_type = RepoClientType.READ
        tools.repo_managers = {}

        # Mock get_repo_client to return our test repo clients
        def mock_get_repo_client(repo_name, type):
            return repo_clients[repo_name]

        context.get_repo_client = MagicMock(side_effect=mock_get_repo_client)

        # Create mock repo manager instances
        mock_repo_manager_instances = []
        for _ in range(len(repo_names)):
            manager = MagicMock()
            manager.initialize_in_background = MagicMock()
            mock_repo_manager_instances.append(manager)

        # Make the RepoManager mock return our instances
        mock_repo_manager.side_effect = mock_repo_manager_instances

        # Call _download_repos directly
        tools._download_repos()

        # Verify that get_repo_client was called for each repo with correct args
        assert context.get_repo_client.call_count == len(repo_names)
        for repo_name in repo_names:
            context.get_repo_client.assert_any_call(repo_name=repo_name, type=RepoClientType.READ)

        # Verify that the repo_managers dict was populated
        assert len(tools.repo_managers) == len(repo_names)

        # Verify that initialize_in_background was called for each manager
        for manager in mock_repo_manager_instances:
            manager.initialize_in_background.assert_called_once()

    @patch("seer.automation.autofix.tools.tools.RepoManager")
    def test_download_repos_empty(self, mock_repo_manager):
        """Test that _download_repos handles empty repo lists correctly."""
        # Setup
        context = MagicMock()

        # Create the tools instance without mocking _download_repos
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=[])  # Empty repo list
        tools.retrieval_top_k = 8
        tools.repo_client_type = RepoClientType.READ
        tools.repo_managers = {}

        # Call _download_repos directly
        tools._download_repos()

        # Verify that get_repo_client was not called
        context.get_repo_client.assert_not_called()

        # Verify that RepoManager was not created
        mock_repo_manager.assert_not_called()

        # Verify that the repo_managers dict remains empty
        assert len(tools.repo_managers) == 0

    @patch("seer.automation.autofix.tools.tools.RepoManager")
    def test_download_repos_exception(self, mock_repo_manager):
        """Test that _download_repos handles exceptions during repo client creation."""
        # Setup
        context = MagicMock()
        repo_names = ["owner/repo1", "owner/repo2", "owner/error_repo"]

        # Create the tools instance without mocking _download_repos
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=repo_names)
        tools.retrieval_top_k = 8
        tools.repo_client_type = RepoClientType.READ
        tools.repo_managers = {}

        # Mock get_repo_client to raise an exception for one repo
        def mock_get_repo_client(repo_name, type):
            if repo_name == "owner/error_repo":
                raise ValueError("Error creating repo client")
            return MagicMock()

        context.get_repo_client = MagicMock(side_effect=mock_get_repo_client)

        # Create mock repo manager instances
        mock_repo_manager_instances = []
        for _ in range(2):  # Only two successful repos
            manager = MagicMock()
            manager.initialize_in_background = MagicMock()
            mock_repo_manager_instances.append(manager)

        # Make the RepoManager mock return our instances
        mock_repo_manager.side_effect = mock_repo_manager_instances

        # Call _download_repos directly - should raise an exception
        with pytest.raises(ValueError):
            tools._download_repos()

        # Verify that get_repo_client was called for each repo until exception
        assert context.get_repo_client.call_count >= 2  # At least the first two repos


class TestEnsureReposDownloaded:
    @patch("seer.automation.autofix.tools.tools.wait")
    def test_ensure_repos_downloaded_specific_repo(self, mock_wait):
        """Test that _ensure_repos_downloaded only waits for a specific repo when repo_name is provided."""
        # Setup
        context = MagicMock()
        context.event_manager = MagicMock()

        repo_names = ["owner/repo1", "owner/repo2", "owner/repo3"]
        specific_repo_name = "owner/repo2"

        # Create mock repo managers with initialization futures
        repo_managers = {}
        futures = {}

        for name in repo_names:
            future = MagicMock(spec=Future)
            future.exception.return_value = None

            repo_manager = MagicMock()
            repo_manager.initialization_future = future
            repo_manager.repo_client.repo_full_name = name
            repo_manager.is_available = False
            repo_manager.is_cancelled = False  # Explicitly set is_cancelled to False

            repo_managers[name] = repo_manager
            futures[name] = future

        # Mock wait to return done and not_done sets
        mock_wait.return_value = (set([futures[specific_repo_name]]), set())

        # Create the tools instance
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=repo_names)
        tools.repo_managers = repo_managers
        tools._trigger_liveness_probe = MagicMock()

        # Patch the internal wait function to use our mock
        with patch("seer.automation.autofix.tools.tools.append_langfuse_observation_metadata"):
            with patch("seer.automation.autofix.tools.tools.time"):
                # Call _ensure_repos_downloaded with a specific repo
                tools._ensure_repos_downloaded(specific_repo_name)

        # Verify that wait was called with only the future for the specific repo
        mock_wait.assert_called_once()
        args, kwargs = mock_wait.call_args
        assert len(args[0]) == 1
        assert args[0][0] == futures[specific_repo_name]
        assert kwargs["timeout"] == 240.0
        assert kwargs["return_when"] == FIRST_EXCEPTION

        # Verify that event_manager.add_log was called
        context.event_manager.add_log.assert_called_once()

    @patch("seer.automation.autofix.tools.tools.wait")
    def test_ensure_repos_downloaded_all_repos(self, mock_wait):
        """Test that _ensure_repos_downloaded waits for all repos when repo_name is None."""
        # Setup
        context = MagicMock()
        context.event_manager = MagicMock()

        repo_names = ["owner/repo1", "owner/repo2", "owner/repo3"]

        # Create mock repo managers with initialization futures
        repo_managers = {}
        all_futures = []

        for name in repo_names:
            future = MagicMock(spec=Future)
            future.exception.return_value = None

            repo_manager = MagicMock()
            repo_manager.initialization_future = future
            repo_manager.repo_client.repo_full_name = name
            repo_manager.is_available = False
            repo_manager.is_cancelled = False  # Explicitly set is_cancelled to False

            repo_managers[name] = repo_manager
            all_futures.append(future)

        # Mock wait to return all futures as done
        mock_wait.return_value = (set(all_futures), set())

        # Create the tools instance
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=repo_names)
        tools.repo_managers = repo_managers
        tools._trigger_liveness_probe = MagicMock()

        # Patch the internal wait function to use our mock
        with patch("seer.automation.autofix.tools.tools.append_langfuse_observation_metadata"):
            with patch("seer.automation.autofix.tools.tools.time"):
                # Call _ensure_repos_downloaded without specifying a repo
                tools._ensure_repos_downloaded()

        # Verify that wait was called with all futures
        mock_wait.assert_called_once()
        args, kwargs = mock_wait.call_args
        assert len(args[0]) == 3
        assert set(args[0]) == set(all_futures)

        # Verify that event_manager.add_log was called
        context.event_manager.add_log.assert_called_once()

    @patch("seer.automation.autofix.tools.tools.wait")
    def test_ensure_repos_downloaded_already_downloaded(self, mock_wait):
        """Test that _ensure_repos_downloaded handles repos that are already downloaded."""
        # Setup
        context = MagicMock()
        context.event_manager = MagicMock()

        repo_names = ["owner/repo1", "owner/repo2"]

        # Create mock repo managers with no initialization futures (already downloaded)
        repo_managers = {}

        for name in repo_names:
            repo_manager = MagicMock()
            repo_manager.initialization_future = None  # Already downloaded
            repo_manager.repo_client.repo_full_name = name
            repo_manager.is_available = True
            repo_manager.is_cancelled = False  # Explicitly set is_cancelled to False

            repo_managers[name] = repo_manager

        # Create the tools instance
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=repo_names)
        tools.repo_managers = repo_managers

        # Call _ensure_repos_downloaded
        tools._ensure_repos_downloaded()

        # Verify that wait was not called (no futures to wait for)
        mock_wait.assert_not_called()

        # Verify that event_manager.add_log was not called (no need to wait)
        context.event_manager.add_log.assert_not_called()

    @patch("seer.automation.autofix.tools.tools.wait")
    def test_ensure_repos_downloaded_timeout(self, mock_wait):
        """Test that _ensure_repos_downloaded handles timeouts correctly."""
        # Setup
        context = MagicMock()
        context.event_manager = MagicMock()

        repo_names = ["owner/repo1", "owner/repo2"]

        # Create mock repo managers with initialization futures
        repo_managers = {}
        futures = {}

        for name in repo_names:
            future = MagicMock(spec=Future)
            future.exception.return_value = None

            repo_manager = MagicMock()
            repo_manager.initialization_future = future
            repo_manager.repo_client.repo_full_name = name
            repo_manager.mark_as_timed_out = MagicMock()
            repo_manager.is_available = False
            repo_manager.is_cancelled = False  # Explicitly set is_cancelled to False

            repo_managers[name] = repo_manager
            futures[name] = future

        # Mock wait to return one future as not done (timed out)
        mock_wait.return_value = (
            set([futures["owner/repo1"]]),  # done
            set([futures["owner/repo2"]]),  # not done (timed out)
        )

        # Create the tools instance
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=repo_names)
        tools.repo_managers = repo_managers
        tools._trigger_liveness_probe = MagicMock()

        # Patch the internal wait function to use our mock
        with patch("seer.automation.autofix.tools.tools.append_langfuse_observation_metadata"):
            with patch("seer.automation.autofix.tools.tools.time"):
                # Call _ensure_repos_downloaded
                tools._ensure_repos_downloaded()

        # Verify that wait was called with all futures
        mock_wait.assert_called_once()

        # Verify that mark_as_timed_out was called for the timed out repo
        repo_managers["owner/repo2"].mark_as_timed_out.assert_called_once()
        repo_managers["owner/repo1"].mark_as_timed_out.assert_not_called()

    @patch("seer.automation.autofix.tools.tools.wait")
    def test_ensure_repos_downloaded_exception(self, mock_wait):
        """Test that _ensure_repos_downloaded handles exceptions from initialization_future."""
        # Setup
        context = MagicMock()
        context.event_manager = MagicMock()

        repo_names = ["owner/repo1", "owner/repo2"]

        # Create mock repo managers with initialization futures
        repo_managers = {}

        # First repo has a successful future
        future1 = MagicMock(spec=Future)
        future1.exception.return_value = None

        # Second repo has a future with an exception
        future2 = MagicMock(spec=Future)
        future2.exception.return_value = ValueError("Failed to initialize")

        repo_manager1 = MagicMock()
        repo_manager1.initialization_future = future1
        repo_manager1.repo_client.repo_full_name = "owner/repo1"
        repo_manager1.mark_as_timed_out = MagicMock()
        repo_manager1.is_available = False
        repo_manager1.is_cancelled = False  # Explicitly set is_cancelled to False

        repo_manager2 = MagicMock()
        repo_manager2.initialization_future = future2
        repo_manager2.repo_client.repo_full_name = "owner/repo2"
        repo_manager2.mark_as_timed_out = MagicMock()
        repo_manager2.is_available = False
        repo_manager2.is_cancelled = False  # Explicitly set is_cancelled to False

        repo_managers["owner/repo1"] = repo_manager1
        repo_managers["owner/repo2"] = repo_manager2

        # Mock wait to return all futures as done
        mock_wait.return_value = (set([future1, future2]), set())

        # Create the tools instance
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=repo_names)
        tools.repo_managers = repo_managers
        tools._trigger_liveness_probe = MagicMock()

        # Patch the internal wait function to use our mock
        with patch("seer.automation.autofix.tools.tools.append_langfuse_observation_metadata"):
            with patch("seer.automation.autofix.tools.tools.time"):
                # Call _ensure_repos_downloaded
                tools._ensure_repos_downloaded()

        # Verify that mark_as_timed_out was called for the repo with exception
        repo_managers["owner/repo2"].mark_as_timed_out.assert_called_once()
        repo_managers["owner/repo1"].mark_as_timed_out.assert_not_called()

    @patch("seer.automation.autofix.tools.tools.wait")
    def test_ensure_repos_downloaded_skips_cancelled_repos(self, mock_wait):
        """Test that _ensure_repos_downloaded skips cancelled repos."""
        # Setup
        context = MagicMock()
        context.event_manager = MagicMock()

        # Create mock repo managers
        repo_managers = {"normal/repo": MagicMock(), "cancelled/repo": MagicMock()}

        # Set up a normal repo
        normal_repo = repo_managers["normal/repo"]
        normal_repo.is_available = False
        normal_repo.is_cancelled = False
        normal_repo.repo_client.repo_full_name = "normal/repo"
        normal_future = MagicMock(spec=Future)
        normal_repo.initialization_future = normal_future

        # Set up a cancelled repo
        cancelled_repo = repo_managers["cancelled/repo"]
        cancelled_repo.is_available = False
        cancelled_repo.is_cancelled = True
        cancelled_repo.repo_client.repo_full_name = "cancelled/repo"
        cancelled_future = MagicMock(spec=Future)
        cancelled_repo.initialization_future = cancelled_future

        # Mock wait to return done and not_done sets
        mock_wait.return_value = (set([normal_future]), set())

        # Create the tools instance
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=["normal/repo", "cancelled/repo"])
        tools.repo_managers = repo_managers
        tools._trigger_liveness_probe = MagicMock()

        # Patch langfuse and time
        with patch("seer.automation.autofix.tools.tools.append_langfuse_observation_metadata"):
            with patch("seer.automation.autofix.tools.tools.time"):
                # Call _ensure_repos_downloaded with no specific repo
                tools._ensure_repos_downloaded()

        # Verify that wait was called with only the normal repo's future
        mock_wait.assert_called_once()
        args, kwargs = mock_wait.call_args
        assert len(args[0]) == 1
        assert args[0][0] == normal_future
        assert cancelled_future not in args[0]
        assert kwargs["timeout"] == 240.0
        assert kwargs["return_when"] == FIRST_EXCEPTION

    @patch("seer.automation.autofix.tools.tools.wait")
    def test_ensure_specific_repo_downloaded_skips_if_cancelled(self, mock_wait):
        """Test that _ensure_repos_downloaded skips a specific repo if it's cancelled."""
        # Setup
        context = MagicMock()
        context.event_manager = MagicMock()

        # Create mock repo managers
        repo_managers = {"cancelled/repo": MagicMock()}

        # Set up a cancelled repo
        cancelled_repo = repo_managers["cancelled/repo"]
        cancelled_repo.is_available = False
        cancelled_repo.is_cancelled = True
        cancelled_repo.repo_client.repo_full_name = "cancelled/repo"
        cancelled_future = MagicMock(spec=Future)
        cancelled_repo.initialization_future = cancelled_future

        # Create the tools instance
        tools = BaseTools.__new__(BaseTools)
        tools.context = context
        tools._get_repo_names = MagicMock(return_value=["cancelled/repo"])
        tools.repo_managers = repo_managers
        tools._trigger_liveness_probe = MagicMock()

        # Call _ensure_repos_downloaded for the cancelled repo
        tools._ensure_repos_downloaded("cancelled/repo")

        # Verify wait was not called at all because the repo is cancelled
        mock_wait.assert_not_called()

        # Verify event_manager.add_log was not called
        context.event_manager.add_log.assert_not_called()


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
            == "Could not find the file in the given commit. Either your hash/SHA is incorrect (it must be the 7 character SHA, found through explain_file), or the file does not exist in the given commit."
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
        autofix_tools.repo_managers = {repo_name: MagicMock(is_available=False)}

        result = autofix_tools.run_ripgrep(query="foo", repo_name=repo_name)

        assert "Error: We had an issue loading the repository" in result
        autofix_tools._ensure_repos_downloaded.assert_called_once_with(repo_name)

    @patch("seer.automation.autofix.tools.tools.run_ripgrep_in_repo")
    def test_single_repo_success(self, mock_run_ripgrep, autofix_tools: BaseTools):
        # Setup
        repo_name = "test/repo"
        autofix_tools._get_repo_names = MagicMock(return_value=[repo_name])
        autofix_tools._ensure_repos_downloaded = MagicMock()
        autofix_tools.repo_managers = {
            repo_name: MagicMock(is_available=True, repo_path="/tmp/foo")
        }
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
        autofix_tools.repo_managers = {repo_name: MagicMock(is_available=True, repo_path="/tmp")}
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
        autofix_tools.repo_managers = {
            "owner/repo1": MagicMock(is_available=True, repo_path="/d1"),
            "owner/repo2": MagicMock(is_available=True, repo_path="/d2"),
            "owner/repo3": MagicMock(is_available=False),
        }

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
        assert "Result for owner/repo1:\nA\nResult for owner/repo2:\nB" in result
        assert "Result for owner/repo3:\nError:" not in result

    @patch("seer.automation.autofix.tools.tools.run_ripgrep_in_repo")
    def test_no_results_found(self, mock_run_ripgrep, autofix_tools: BaseTools):
        # Setup
        repo_name = "test/repo"
        autofix_tools._get_repo_names = MagicMock(return_value=[repo_name])
        autofix_tools._ensure_repos_downloaded = MagicMock()
        autofix_tools.repo_managers = {
            repo_name: MagicMock(is_available=True, repo_path="/tmp/foo")
        }
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
        autofix_tools.repo_managers = {
            "owner/test_repo": MagicMock(is_available=True, repo_path="/tmp/test_dir/repo")
        }
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
        autofix_tools.repo_managers = {
            "owner/test_repo": MagicMock(is_available=True, repo_path="/tmp/test_dir/repo")
        }
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        result = autofix_tools.find_files("find . -name '*.nonexistent'")

        assert "Results from owner/test_repo: no files found." in result

    @patch("subprocess.run")
    def test_find_files_error(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.repo_managers = {
            "owner/test_repo": MagicMock(is_available=True, repo_path="/tmp/test_dir/repo")
        }
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "find: invalid option -- z"
        mock_run.return_value = mock_process

        result = autofix_tools.find_files("find . -z '*.py'")

        assert "Results from owner/test_repo: find: invalid option -- z" in result

    @patch("subprocess.run")
    def test_find_files_multiple_repos(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.repo_managers = {
            "owner/repo1": MagicMock(is_available=True, repo_path="/tmp/test_dir1/repo"),
            "owner/repo2": MagicMock(is_available=True, repo_path="/tmp/test_dir2/repo"),
            "owner/repo3": MagicMock(is_available=False),
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
        assert "Error: We had an issue loading the repository `owner/repo3`" not in result

    @patch("seer.automation.autofix.tools.tools.BaseTools._ensure_repos_downloaded")
    def test_find_files_specific_repo(self, mock_ensure_repos, autofix_tools: BaseTools):
        autofix_tools.repo_managers = {
            "owner/test_repo": MagicMock(is_available=True, repo_path="/tmp/test_dir/repo")
        }
        autofix_tools.context.autocorrect_repo_name = MagicMock(return_value="owner/test_repo")
        with patch("subprocess.run") as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "./file.py"
            mock_run.return_value = mock_process

            autofix_tools.find_files("find . -name '*.py'", repo_name="owner/test_repo")

            mock_ensure_repos.assert_called_once_with("owner/test_repo")
            mock_run.assert_called_once()


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
        mock_repo_client = MagicMock()
        mock_repo_client.does_file_exist.return_value = False
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.does_file_exist.return_value = False
        kwargs = {"file_text": "new file content"}

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

        mock_state = MagicMock(request=MagicMock(spec=AutofixRequest))
        mock_state.codebases = {
            "123": mock_codebase,
        }
        mock_state.request.repos = [mock_repo]
        mock_state.readable_repos = [mock_repo]

        autofix_tools.context.state.get.return_value = mock_state

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

        mock_state = MagicMock(request=MagicMock(spec=AutofixRequest))
        mock_state.request.repos = []  # Set empty repos to trigger "No file changes found to undo"

        autofix_tools.context.state.get.return_value = mock_state

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
        mock_repo_client = MagicMock()
        mock_repo_client.get_valid_file_paths.return_value = set()
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.state.get.return_value = MagicMock(
            request=MagicMock(codebases={"123": MagicMock(file_changes=[])}),
            readable_repos=[MagicMock(full_name="test/repo")],
        )

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
        mock_repo_client = MagicMock()
        mock_repo_client.does_file_exist.return_value = False
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.does_file_exist.return_value = False
        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = repo_name
        # Mock _get_repo_names to return a single repo
        autofix_tools.context._get_repo_names = MagicMock(return_value=[repo_name])
        # Mock _attempt_fix_path to return None, simulating the path not existing
        # Note: This mock is on the context, as that's where _get_repo_name_and_path calls it
        autofix_tools._attempt_fix_path = MagicMock(return_value=None)
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
            autofix_tools._attempt_fix_path.assert_called_once_with(
                new_path, repo_name, ignore_local_changes=False
            )
            autofix_tools.context.does_file_exist.assert_called_once_with(
                path=new_path, repo_name=repo_name, ignore_local_changes=False
            )
            mock_make_patches.assert_called_once()
            autofix_tools._append_file_change.assert_called_once()
            assert "Change applied successfully" in result
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
        autofix_tools._attempt_fix_path.assert_called_once_with(
            nonexistent_path, repo_name, ignore_local_changes=False
        )
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
        autofix_tools.repo_managers = {
            repo_name: MagicMock(is_available=True, repo_path="/tmp/test/repo")
        }

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
        autofix_tools.repo_managers = {
            repo_name: MagicMock(is_available=True, repo_path="/tmp/test/repo")
        }

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
        autofix_tools.repo_managers = {
            repo_name: MagicMock(is_available=False),
        }

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
            assert "Error: We had an issue loading the repository" in result

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

    def test_expand_document_with_invalid_path(self, autofix_tools: BaseTools):
        """Test that expand_document returns an error when the path is invalid"""
        # Setup
        repo_name = "test/repo"
        invalid_path = "invalid/path.py"

        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = repo_name

        # Mock attempt_fix_path to return None (indicating invalid path)
        autofix_tools._attempt_fix_path = MagicMock(return_value=None)

        # Call expand_document
        result = autofix_tools.expand_document(invalid_path, repo_name)

        # Verify behavior
        autofix_tools.context.autocorrect_repo_name.assert_called_once_with(repo_name)
        autofix_tools._attempt_fix_path.assert_has_calls(
            [
                call(invalid_path, repo_name, files_only=True),
                call(invalid_path, repo_name, files_only=False),
            ]
        )

        # Assert that an error message is returned
        assert "Error: The file path" in result
        assert "doesn't exist" in result
        assert invalid_path in result

    def test_incorrect_repo_name_returns_error(self, autofix_tools: BaseTools):
        """Test that expand_document returns an error when the repo name is incorrect"""
        # Setup
        repo_name = "test/repo"
        file_path = "src/test.py"

        # Mock autocorrect_repo_name to return the repo name
        autofix_tools.context.autocorrect_repo_name.return_value = None

        # Create a mock state with the valid repo
        valid_repo = RepoDefinition(
            provider="github",
            owner="valid",
            name="repo",
            external_id="123",
        )
        mock_state = MagicMock()
        mock_state.request.repos = [valid_repo]
        mock_state.readable_repos = [valid_repo]
        autofix_tools.context.state.get.return_value = mock_state

        result = autofix_tools.expand_document(file_path, repo_name)

        # Assert
        assert "Error: Repo 'test/repo' not found" in result
        assert "valid/repo" in result
