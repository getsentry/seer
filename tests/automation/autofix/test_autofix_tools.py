import textwrap
from unittest.mock import MagicMock, patch

import pytest

from seer.automation.agent.client import LlmClient
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType


@pytest.fixture
def autofix_tools():
    context = MagicMock(spec=AutofixContext)
    context.event_manager = MagicMock()
    context.state = MagicMock()
    context.state.get.return_value.readable_repos = []
    with patch(
        "seer.automation.autofix.tools.BaseTools._start_parallel_repo_download", MagicMock()
    ):
        tools = BaseTools(context)
    return tools


class TestFileSystem:
    @patch("seer.automation.autofix.tools.cleanup_dir")
    def test_context_manager_cleanup(self, mock_cleanup_dir, autofix_tools: BaseTools):
        with autofix_tools as tools:
            tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert autofix_tools.tmp_dir is None

    @patch("seer.automation.autofix.tools.cleanup_dir")
    def test_cleanup_method(self, mock_cleanup_dir, autofix_tools: BaseTools):
        autofix_tools.tmp_dir = {"dummy": ("/tmp/test_dir", "/tmp/test_dir/repo")}

        autofix_tools.cleanup()

        mock_cleanup_dir.assert_called_once_with("/tmp/test_dir")
        assert autofix_tools.tmp_dir is None

    @patch("seer.automation.autofix.tools.cleanup_dir")
    def test_cleanup_not_called_when_tmp_dir_is_none(
        self, mock_cleanup_dir, autofix_tools: BaseTools
    ):
        assert autofix_tools.tmp_dir == {}

        autofix_tools.cleanup()

        mock_cleanup_dir.assert_not_called()
        assert autofix_tools.tmp_dir is None


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


class TestParallelRepoDownload:
    @patch("seer.automation.autofix.tools.copy_modules_initializer")
    @patch("seer.automation.autofix.tools.ThreadPoolExecutor")
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

    @patch("seer.automation.autofix.tools.ThreadPoolExecutor")
    def test_cleanup_cancels_download_future(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor
        mock_future = MagicMock()

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = [MagicMock(full_name="owner/repo1")]

        # Patch _start_parallel_repo_download to avoid starting real threads
        with patch("seer.automation.autofix.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Set a mock future
        tools._download_future = mock_future
        tools._download_future.done.return_value = False

        # Call cleanup
        tools.cleanup()

        # Verify future was cancelled
        mock_future.cancel.assert_called_once()
        assert tools._download_future is None

    @patch("seer.automation.autofix.tools.ThreadPoolExecutor")
    def test_exit_shuts_down_executor(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = []

        # Patch _start_parallel_repo_download to avoid starting real threads
        with patch("seer.automation.autofix.tools.BaseTools._start_parallel_repo_download"):
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
        with patch("seer.automation.autofix.tools.BaseTools._start_parallel_repo_download"):
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

    @patch("seer.automation.autofix.tools.ThreadPoolExecutor")
    @patch("seer.automation.autofix.tools.as_completed")
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
        with patch("seer.automation.autofix.tools.BaseTools._start_parallel_repo_download"):
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
        with patch("seer.automation.autofix.tools.append_langfuse_observation_metadata"):
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

    @patch("seer.automation.autofix.tools.ThreadPoolExecutor")
    def test_cleanup_handles_future_cancel_exception(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = []

        # Create tools
        with patch("seer.automation.autofix.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Set up a future that raises an exception when cancelled
        mock_future = MagicMock()
        mock_future.done.return_value = False
        mock_future.cancel.side_effect = Exception("Cancel failed")
        tools._download_future = mock_future

        # Call cleanup - should handle the exception gracefully
        with patch("seer.automation.autofix.tools.logger.exception") as mock_logger:
            tools.cleanup()

        # Verify exception was logged
        mock_logger.assert_called_once()
        assert "Cancel failed" in str(mock_logger.call_args)
        assert tools._download_future is None

    @patch("seer.automation.autofix.tools.ThreadPoolExecutor")
    def test_exit_handles_executor_shutdown_exception(self, mock_executor_class):
        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor
        mock_executor.shutdown.side_effect = Exception("Shutdown failed")

        context = MagicMock(spec=AutofixContext)
        context.state = MagicMock()
        context.state.get.return_value.readable_repos = []

        # Create tools
        with patch("seer.automation.autofix.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Call __exit__ - should handle the exception gracefully
        with patch("seer.automation.autofix.tools.logger.exception") as mock_logger:
            tools.__exit__(None, None, None)

        # Verify exception was logged
        mock_logger.assert_called_once()
        assert "Shutdown failed" in str(mock_logger.call_args)
        mock_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    @patch("seer.automation.autofix.tools.ThreadPoolExecutor")
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
        with patch("seer.automation.autofix.tools.BaseTools._start_parallel_repo_download"):
            tools = BaseTools(context)

        # Set up tmp_dir to simulate already downloaded repo
        tools.tmp_dir = {"owner/repo1": ("/tmp/dir1", "/tmp/dir1/repo")}
        tools._get_repo_names = MagicMock(return_value=["owner/repo1", "owner/repo2"])

        # Mock repo client
        mock_repo_client = MagicMock()
        mock_repo_client.load_repo_to_tmp_dir.return_value = ("/tmp/dir2", "/tmp/dir2/repo")
        context.get_repo_client.return_value = mock_repo_client

        # Call method
        with patch("seer.automation.autofix.tools.append_langfuse_observation_metadata"):
            tools._ensure_repos_downloaded()

        # Verify only the second repo was downloaded
        context.get_repo_client.assert_called_once_with(
            repo_name="owner/repo2", type=tools.repo_client_type
        )
        mock_repo_client.load_repo_to_tmp_dir.assert_called_once()
        assert "owner/repo1" in tools.tmp_dir
        assert "owner/repo2" in tools.tmp_dir
