import textwrap
from unittest.mock import MagicMock, patch

import pytest

from seer.automation.agent.client import LlmClient
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.tools import BaseTools
from seer.automation.codebase.repo_client import RepoClientType


@pytest.fixture
def autofix_tools():
    context = MagicMock(AutofixContext)
    context.event_manager = MagicMock()
    context.state = MagicMock()
    return BaseTools(context)


class TestFileSearch:
    def test_file_search_found(self, autofix_tools: BaseTools):
        # Set up a dummy repo so that _get_repo_names() returns "testowner/testrepo"
        dummy_repo = MagicMock(full_name="testowner/testrepo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/file2.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search("file2.py")
        # Expected formatting: a newline, header with repo name, then sorted matches (note that
        # "src/subfolder/file2.py" comes before "tests/file2.py" alphabetically).
        expected = (
            "\n FILES IN REPO testowner/testrepo:\n  src/subfolder/file2.py\n  tests/file2.py"
        )
        assert result == expected

    def test_file_search_not_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="testowner/testrepo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/file2.py",
            "src/subfolder/file3.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search("nonexistent.py")
        expected = "no file with name nonexistent.py found in any repository"
        assert result == expected

    def test_file_search_with_repo_name(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {"src/file1.py"}
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.repo_client_type = RepoClientType.READ

        autofix_tools.file_search("file1.py")
        autofix_tools.context.get_repo_client.assert_any_call(
            repo_name="owner/test_repo", type=RepoClientType.READ
        )

    def test_file_search_multi_repo_overlapping(self, autofix_tools: BaseTools):
        # Set up two dummy repositories with overlapping filenames
        dummy_repo1 = MagicMock(full_name="ownerA/repoA")
        dummy_repo2 = MagicMock(full_name="ownerB/repoB")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo1, dummy_repo2]

        # Prepare repo clients with overlapping file names
        client_repo1 = MagicMock()
        client_repo1.get_index_file_set.return_value = {"common/overlap.py", "unique/file1.py"}
        client_repo2 = MagicMock()
        client_repo2.get_index_file_set.return_value = {"lib/overlap.py", "unique/file2.py"}

        # Return a specific client based on the repository name
        def get_repo_client_side_effect(repo_name, type):
            if repo_name == "ownerA/repoA":
                return client_repo1
            elif repo_name == "ownerB/repoB":
                return client_repo2
            return MagicMock()

        autofix_tools.context.get_repo_client.side_effect = get_repo_client_side_effect

        result = autofix_tools.file_search("overlap.py")
        expected = (
            "\n FILES IN REPO ownerA/repoA:\n  common/overlap.py"
            "\n FILES IN REPO ownerB/repoB:\n  lib/overlap.py"
        )
        assert result == expected

    def test_file_search_multi_repo(self, autofix_tools: BaseTools):
        # Set up two dummy repositories.
        dummy_repo1 = MagicMock(full_name="owner/repo1")
        dummy_repo2 = MagicMock(full_name="owner/repo2")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo1, dummy_repo2]

        client_repo1 = MagicMock()
        client_repo1.get_index_file_set.return_value = {
            "src/app/file.py",
            "docs/file.py",
            "README.md",
        }
        client_repo2 = MagicMock()
        client_repo2.get_index_file_set.return_value = {
            "src/file.py",
            "file.py",
            "test/test_file.py",
        }

        # Return the appropriate repo client based on the repository name.
        def get_repo_client_side_effect(repo_name, type):
            if repo_name == "owner/repo1":
                return client_repo1
            elif repo_name == "owner/repo2":
                return client_repo2
            return MagicMock()

        autofix_tools.context.get_repo_client.side_effect = get_repo_client_side_effect

        result = autofix_tools.file_search("file.py")
        expected = (
            "\n FILES IN REPO owner/repo1:\n  docs/file.py\n  src/app/file.py"
            "\n FILES IN REPO owner/repo2:\n  file.py\n  src/file.py"
        )
        assert result == expected


class TestFileSearchWildcard:
    def test_file_search_wildcard_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="testowner/testrepo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search_wildcard("*.py")
        expected = "\n FILES IN REPO testowner/testrepo:\n  src/file1.py\n  src/subfolder/file2.py\n  tests/test_file1.py"
        assert result == expected

    def test_file_search_wildcard_not_found(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="testowner/testrepo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {
            "src/file1.py",
            "tests/test_file1.py",
            "src/subfolder/file2.py",
        }
        autofix_tools.context.get_repo_client.return_value = mock_repo_client

        result = autofix_tools.file_search_wildcard("*.js")
        expected = "No files matching pattern '*.js' found in any repository"
        assert result == expected

    def test_file_search_wildcard_with_repo_name(self, autofix_tools: BaseTools):
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        mock_repo_client = MagicMock()
        mock_repo_client.get_index_file_set.return_value = {"src/file1.py"}
        autofix_tools.context.get_repo_client.return_value = mock_repo_client
        autofix_tools.repo_client_type = RepoClientType.READ

        autofix_tools.file_search_wildcard("*.py")
        autofix_tools.context.get_repo_client.assert_any_call(
            repo_name="owner/test_repo", type=RepoClientType.READ
        )

    def test_file_search_wildcard_multi_repo(self, autofix_tools: BaseTools):
        # Set up two dummy repositories.
        dummy_repo1 = MagicMock(full_name="owner/repo1")
        dummy_repo2 = MagicMock(full_name="owner/repo2")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo1, dummy_repo2]

        # Prepare each repo client with its own file index.
        client_repo1 = MagicMock()
        client_repo1.get_index_file_set.return_value = {"src/app/file.py", "docs/readme.txt"}
        client_repo2 = MagicMock()
        client_repo2.get_index_file_set.return_value = {"src/file.py", "test/test_file.py"}

        # Return the appropriate repo client based on the repository name.
        def get_repo_client_side_effect(repo_name, type):
            if repo_name == "owner/repo1":
                return client_repo1
            elif repo_name == "owner/repo2":
                return client_repo2
            return MagicMock()

        autofix_tools.context.get_repo_client.side_effect = get_repo_client_side_effect

        result = autofix_tools.file_search_wildcard("*file.py")
        expected = (
            "\n FILES IN REPO owner/repo1:\n  src/app/file.py"
            "\n FILES IN REPO owner/repo2:\n  src/file.py\n  test/test_file.py"
        )
        assert result == expected


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


class TestKeywordSearch:
    def test_keyword_search_found(self, autofix_tools: BaseTools):
        # Set up a dummy repository
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        # Prepare a dummy repo client with a temporary directory
        dummy_repo_client = MagicMock()
        dummy_repo_client.load_repo_to_tmp_dir.return_value = (
            "/tmp/test_dir",
            "/tmp/test_repo_dir",
        )
        autofix_tools.context.get_repo_client.return_value = dummy_repo_client

        # Patch CodeSearcher and MatchXml to simulate search results
        with (
            patch("seer.automation.autofix.tools.CodeSearcher") as mock_searcher_class,
            patch("seer.automation.autofix.tools.MatchXml") as mock_matchxml,
        ):

            fake_searcher = MagicMock()
            fake_result = MagicMock()
            fake_result.relative_path = "src/example.py"
            fake_match = MagicMock()
            fake_match.context = "fake context snippet"
            fake_result.matches = [fake_match]
            fake_searcher.search.return_value = [fake_result]
            mock_searcher_class.return_value = fake_searcher

            fake_matchxml_instance = MagicMock()
            fake_matchxml_instance.to_prompt_str.return_value = "Fake prompt string"
            mock_matchxml.return_value = fake_matchxml_instance

            result = autofix_tools.keyword_search("keyword", supported_extensions=[".py"])

        # Assert that the expected log message was added and the result is as expected.
        autofix_tools.context.event_manager.add_log.assert_called_with(
            "Searched codebase for `keyword`, found 1 result(s)."
        )
        assert result == "Fake prompt string"

    def test_keyword_search_no_results(self, autofix_tools: BaseTools):
        # Set up a dummy repository
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        # Prepare a dummy repo client with a temporary directory
        dummy_repo_client = MagicMock()
        dummy_repo_client.load_repo_to_tmp_dir.return_value = (
            "/tmp/test_dir",
            "/tmp/test_repo_dir",
        )
        autofix_tools.context.get_repo_client.return_value = dummy_repo_client

        # Patch CodeSearcher to return no results
        with patch("seer.automation.autofix.tools.CodeSearcher") as mock_searcher_class:
            fake_searcher = MagicMock()
            fake_searcher.search.return_value = []  # No results found
            mock_searcher_class.return_value = fake_searcher

            result = autofix_tools.keyword_search("nonexistent", supported_extensions=[".py"])

        autofix_tools.context.event_manager.add_log.assert_called_with(
            "Searched codebase for `nonexistent`, found 0 result(s)."
        )
        assert result == "No results found."

    def test_keyword_search_skips_repo(self, autofix_tools: BaseTools):
        # Set up a dummy repository where the repo client's temporary repo directory is falsy
        dummy_repo = MagicMock(full_name="owner/test_repo")
        autofix_tools.context.state.get.return_value.readable_repos = [dummy_repo]

        dummy_repo_client = MagicMock()
        dummy_repo_client.load_repo_to_tmp_dir.return_value = ("/tmp/test_dir", None)
        autofix_tools.context.get_repo_client.return_value = dummy_repo_client

        result = autofix_tools.keyword_search("keyword", supported_extensions=[".py"])

        autofix_tools.context.event_manager.add_log.assert_called_with(
            "Searched codebase for `keyword`, found 0 result(s)."
        )
        assert result == "No results found."


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
