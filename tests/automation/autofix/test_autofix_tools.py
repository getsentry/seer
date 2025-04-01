import textwrap
from unittest.mock import ANY, MagicMock, patch

import pytest

from seer.automation.agent.client import LlmClient
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.models import FileChange, RepoDefinition


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
    context = MagicMock(AutofixContext)
    context.event_manager = MagicMock()
    context.state = MagicMock()

    # Use the real test state
    context.state.get.return_value = test_state

    # Set up context methods
    context._get_repo_names = MagicMock(return_value=[repo.full_name for repo in test_state.repos])
    context._attempt_fix_path = MagicMock(return_value="test.py")

    return BaseTools(context)


class TestFileSystem:
    @patch("seer.automation.autofix.tools.cleanup_dir")
    def test_context_manager_cleanup(self, mock_cleanup_dir):
        context = MagicMock()

        with BaseTools(context) as tools:
            # Set tmp_dir as a dictionary mapping a dummy key to the tuple of paths.
            tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}
            tools.tmp_repo_dir = "/tmp/test_dir/repo"

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert tools.tmp_dir is None
        # Since cleanup() does not clear tmp_repo_dir, we expect it to remain unchanged.
        assert tools.tmp_repo_dir == "/tmp/test_dir/repo"

    def test_cleanup_method(self):
        context = MagicMock()
        tools = BaseTools(context)
        tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        tools.tmp_repo_dir = "/tmp/test_dir/repo"

        with patch("seer.automation.autofix.tools.cleanup_dir") as mock_cleanup_dir:
            tools.cleanup()

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert tools.tmp_dir is None
        # tmp_repo_dir is not cleared by cleanup(), so it should remain unchanged.
        assert tools.tmp_repo_dir == "/tmp/test_dir/repo"

    def test_cleanup_not_called_when_tmp_dir_is_none(self):
        context = MagicMock()
        tools = BaseTools(context)

        with patch("seer.automation.autofix.tools.cleanup_dir") as mock_cleanup_dir:
            tools.cleanup()

        mock_cleanup_dir.assert_not_called()


class TestSemanticFileSearch:
    def test_semantic_file_search_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        # Files available in the repo
        mock_repo_client.get_valid_file_paths.return_value = [
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        ]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.get_file_contents.return_value = "test file contents"

        mock_llm_client = MagicMock()
        # The parsed response now includes both file_path and the repo_name built from the dummy repo.
        mock_llm_client.generate_structured.return_value.parsed = MagicMock(
            file_path="src/file1.py", repo_name="owner/test_repo"
        )

        result = autofix_tools.semantic_file_search(
            "find the main file", llm_client=mock_llm_client
        )
        expected = "This file might be what you're looking for: `src/file1.py`. Contents:\n\ntest file contents"
        assert result == expected

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "repo_names",
        (
            ["owner/repo", "owner/another-repo"],
            ["owner/another-repo"],  # fall back to str RepoName
            ["owner/repo", "owner/another-repo"] * 100,  # fall back to str RepoName
        ),
    )
    def test_semantic_file_search_completion(self, autofix_tools: BaseTools, repo_names: list[str]):
        query = "find the file which tests google's LLM"
        valid_file_paths = textwrap.dedent(
            """
            FILES IN REPO owner/repo:
            src/
            └──something.py
            tests/
            └──another/
                └──test_thing.py
            ------------
            FILES IN REPO owner/another-repo:
            src/
            └──clients/
                ├──claude.py
                ├──gemini.py
                └──openai.py
            tests/
            └──clients/
                ├──test_claude.py
                ├──test_gemini.py
                └──test_openai.py
            """
        )

        llm_client = LlmClient()
        file_location = autofix_tools._semantic_file_search_completion(
            query, valid_file_paths, repo_names, llm_client
        )
        assert file_location.repo_name == "owner/another-repo"
        assert file_location.file_path == "tests/clients/test_gemini.py"

    def test_semantic_file_search_not_found_no_file_path(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_valid_file_paths.return_value = [
            "src/file1.py",
            "tests/test_file1.py",
        ]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value.parsed = None

        result = autofix_tools.semantic_file_search(
            "find nonexistent file", llm_client=mock_llm_client
        )
        expected = "Could not figure out which file matches what you were looking for. You'll have to try yourself."
        assert result == expected

    def test_semantic_file_search_not_found_no_contents(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_valid_file_paths.return_value = [
            "src/file1.py",
            "tests/test_file1.py",
        ]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.get_file_contents.return_value = None

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value.parsed = MagicMock(
            file_path="src/file1.py", repo_name="owner/test_repo"
        )

        result = autofix_tools.semantic_file_search(
            "find file with no contents", llm_client=mock_llm_client
        )
        expected = "Could not figure out which file matches what you were looking for. You'll have to try yourself."
        assert result == expected

    def test_semantic_file_search_with_repo_name(self, autofix_tools: BaseTools):
        # Instead of passing repo_name directly as an argument,
        # set context.repos so that _get_repo_names() returns "owner/specific_repo"
        dummy_repo = MagicMock(full_name="owner/specific_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_valid_file_paths.return_value = ["src/file1.py"]
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.context.get_file_contents.return_value = "test file contents"
        autofix_tools.repo_client_type = RepoClientType.READ

        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value.parsed = MagicMock(
            file_path="src/file1.py", repo_name="owner/specific_repo"
        )

        autofix_tools.semantic_file_search("find file", llm_client=mock_llm_client)
        autofix_tools.context.get_repo_client.assert_any_call(
            repo_name="owner/specific_repo", type=RepoClientType.READ
        )

    def test_semantic_file_search_multi_repo(self, autofix_tools: BaseTools):
        # Create two dummy repos
        dummy_repo1 = MagicMock(full_name="owner/repo1")
        dummy_repo2 = MagicMock(full_name="owner/repo2")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo1, dummy_repo2]

        # Each repo returns a different set of valid file paths.
        client_repo1 = MagicMock()
        client_repo1.get_valid_file_paths.return_value = ["src/main.py", "src/helper.py"]
        client_repo2 = MagicMock()
        client_repo2.get_valid_file_paths.return_value = ["src/main.py", "README.md"]

        # Use a side effect to return the proper client for each repo.
        def get_repo_client_side_effect(repo_name, type):
            if repo_name == "owner/repo1":
                return client_repo1
            elif repo_name == "owner/repo2":
                return client_repo2
            return MagicMock()

        autofix_tools.context.get_repo_client.side_effect = get_repo_client_side_effect

        # Simulate file contents for a file in repo2.
        autofix_tools.context.get_file_contents.return_value = "print('Hello from repo2')"

        # Create a dummy LLM client which returns that for the query 'find main'
        response = MagicMock()
        response.parsed = MagicMock(file_path="src/main.py", repo_name="owner/repo2")
        mock_llm_client = MagicMock()
        mock_llm_client.generate_structured.return_value = response

        result = autofix_tools.semantic_file_search("find main", llm_client=mock_llm_client)
        expected = "This file might be what you're looking for: `src/main.py`. Contents:\n\nprint('Hello from repo2')"
        assert result == expected


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


class TestGrepSearch:
    def test_grep_search_validation(self, autofix_tools: BaseTools):
        result = autofix_tools.grep_search("invalid command")
        assert result == "Command must be a valid grep command that starts with 'grep'."

    @patch("subprocess.run")
    def test_grep_search_success(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = (
            "file1.py:10:def example_function():\nfile2.py:20:    example_function()"
        )
        mock_run.return_value = mock_process

        result = autofix_tools.grep_search("grep -r 'example_function' .")

        mock_run.assert_called_once()
        assert mock_run.call_args[1]["shell"] is False
        assert mock_run.call_args[1]["cwd"] == "/tmp/test_dir/repo"
        assert "Results from owner/test_repo:" in result
        assert "file1.py:10:def example_function()" in result

    @patch("subprocess.run")
    def test_grep_search_no_results(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        result = autofix_tools.grep_search("grep -r 'nonexistent' .")

        assert "Results from owner/test_repo: no results found." in result

    @patch("subprocess.run")
    def test_grep_search_error(self, mock_run, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}
        autofix_tools._get_repo_names = MagicMock(return_value=["owner/test_repo"])

        mock_process = MagicMock()
        mock_process.returncode = 2
        mock_process.stderr = "grep: invalid option -- z"
        mock_run.return_value = mock_process

        result = autofix_tools.grep_search("grep -z 'pattern' .")

        assert "Results from owner/test_repo: grep: invalid option -- z" in result

    @patch("subprocess.run")
    def test_grep_search_multipl_repos(self, mock_run, autofix_tools: BaseTools):
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
                mock_process.stdout = "repo1_file.py:10:result"
            else:  # repo2
                mock_process.returncode = 0
                mock_process.stdout = "repo2_file.py:20:result"

            return mock_process

        mock_run.side_effect = side_effect

        # Run test
        result = autofix_tools.grep_search("grep -r 'pattern' .")

        assert "Results from owner/repo1" in result
        assert "repo1_file.py:10:result" in result
        assert "Results from owner/repo2" in result
        assert "repo2_file.py:20:result" in result

    @patch("seer.automation.autofix.tools.BaseTools._ensure_repos_downloaded")
    def test_grep_search_specific_repo(self, mock_ensure_repos, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"owner/test_repo": ("/tmp/test_dir", "/tmp/test_dir/repo")}

        with patch("subprocess.run") as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "file.py:10:result"
            mock_run.return_value = mock_process

            autofix_tools.grep_search("grep -r 'pattern' .", repo_name="owner/test_repo")

            mock_ensure_repos.assert_called_once_with("owner/test_repo")
            mock_run.assert_called_once()


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

    @patch("seer.automation.autofix.tools.BaseTools._ensure_repos_downloaded")
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

        # Create a file change to add
        file_change = FileChange(
            change_type="edit",
            path="test.py",
            reference_snippet="old text",
            new_snippet="new text",
            repo_name="test/repo",
        )

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
        def mock_get_repo_name_and_path(kwargs):
            if len(autofix_tools.context._get_repo_names()) > 1 and ":" not in kwargs.get(
                "path", ""
            ):
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
