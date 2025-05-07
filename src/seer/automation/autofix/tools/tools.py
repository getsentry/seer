import logging
import os
import shlex
import subprocess
import textwrap
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, as_completed, wait
from typing import Any, Dict, cast

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.tools import ClaudeTool, FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import (
    InsightSharingOutput,
    InsightSharingType,
)
from seer.automation.autofix.models import AutofixRequest
from seer.automation.autofix.tools.read_file_contents import read_file_contents
from seer.automation.autofix.tools.ripgrep_search import run_ripgrep_in_repo
from seer.automation.codebase.file_patches import make_file_patches
from seer.automation.codebase.models import BaseDocument
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codebase.repo_manager import RepoManager
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.models import EventDetails, FileChange, Profile, SentryEventData
from seer.dependency_injection import copy_modules_initializer, inject, injected
from seer.langfuse import append_langfuse_observation_metadata
from seer.rpc import RpcClient

logger = logging.getLogger(__name__)

MAX_FILES_IN_TREE = 100
REPO_WAIT_TIMEOUT_SECS = 120.0


class BaseTools:
    context: AutofixContext | CodegenContext
    retrieval_top_k: int
    repo_managers: Dict[str, RepoManager] = {}
    repo_client_type: RepoClientType = RepoClientType.READ

    def __init__(
        self,
        context: AutofixContext | CodegenContext,
        retrieval_top_k: int = 8,
        repo_client_type: RepoClientType = RepoClientType.READ,
    ):
        self.context = context
        self.retrieval_top_k = retrieval_top_k
        self.repo_client_type = repo_client_type
        self.repo_managers = {}

        self._download_repos()

    def _download_repos(self):
        repo_names = self._get_repo_names()
        if not repo_names:
            return

        for repo_name in repo_names:
            repo_client = self.context.get_repo_client(
                repo_name=repo_name, type=self.repo_client_type
            )
            repo_manager = RepoManager(
                repo_client, trigger_liveness_probe=self._trigger_liveness_probe
            )
            repo_manager.initialize_in_background()
            self.repo_managers[repo_name] = repo_manager

    def _trigger_liveness_probe(self):
        with self.context.state.update() as state:
            # Do nothing, the state should self update updated_at
            pass

    def _ensure_repos_downloaded(self, repo_name: str | None = None):
        """
        Helper method to wait for repos to be downloaded.

        Args:
            repo_name: If provided, only waits for this specific repo to be downloaded.
                      If None, waits for all repos to be downloaded.
        """
        if repo_name:
            repo_names_to_download = [repo_name] if repo_name not in self.repo_managers else []
        else:
            repo_names_to_download = [
                rn for rn in self._get_repo_names() if rn not in self.repo_managers
            ]

        if not repo_names_to_download:
            return

        append_langfuse_observation_metadata({"repo_download": True})

        # Wait for all initialization tasks to complete with a timeout
        start_time = time.time()

        # Collect all futures that need to be waited on
        futures_to_wait = [
            repo_manager.initialization_future
            for repo_manager in self.repo_managers.values()
            if repo_manager.initialization_future
        ]

        self.context.event_manager.add_log(f"Waiting for your repositories to download...")

        # Use concurrent.futures.wait with a single timeout for all futures
        done, not_done = wait(
            futures_to_wait, timeout=REPO_WAIT_TIMEOUT_SECS, return_when=FIRST_EXCEPTION
        )

        # Process results and handle errors
        for repo_manager in self.repo_managers.values():
            if not repo_manager.initialization_future:
                continue

            future = repo_manager.initialization_future
            if future in not_done:
                repo_manager.mark_as_timed_out()
                logger.warning(
                    f"Repository {repo_manager.repo_client.repo_full_name} timed out after {REPO_WAIT_TIMEOUT_SECS} seconds"
                )
            elif future.exception():
                repo_manager.mark_as_timed_out()
                logger.exception(
                    f"Error initializing repository {repo_manager.repo_client.repo_full_name}: {future.exception()}"
                )

        end_time = time.time()
        logger.info(
            f"Repositories became ready in {end_time - start_time} seconds for {len(self.repo_managers)} repositories"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        if hasattr(self, "_executor") and self._executor:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.exception(f"Error shutting down executor: {e}")

    def _get_repo_names(self) -> list[str]:
        if isinstance(self.context, AutofixContext):
            return [repo.full_name for repo in self.context.state.get().readable_repos]
        elif isinstance(self.context, CodegenContext):
            return [self.context.repo.full_name]
        else:
            raise ValueError(f"Unsupported context type: {type(self.context)}")

    def _make_repo_not_found_error_message(self, repo_name: str) -> str:
        return f"Error: Repo '{repo_name}' not found." + (
            f" Available repos: {', '.join([repo.full_name for repo in self.context.repos if isinstance(self.context, AutofixContext)])}"
            if isinstance(self.context, AutofixContext)
            else ""
        )

    def _make_repo_unavailable_error_message(self, repo_name: str) -> str:
        return f"Error: We had an issue loading the repository `{repo_name}`. This tool is unavailable for this repository, you must stop using it for this repository `{repo_name}`."

    @observe(name="Semantic File Search")
    @sentry_sdk.trace
    @inject
    def semantic_file_search(self, query: str, llm_client: LlmClient = injected):
        from seer.automation.autofix.tools.semantic_search import semantic_search

        self.context.event_manager.add_log(f'Searching for "{query}"...')

        result = semantic_search(query=query, context=self.context)

        if not result:
            return "Could not figure out which file matches what you were looking for. You'll have to try yourself."

        return result

    @observe(name="Expand Document")
    @sentry_sdk.trace
    def expand_document(self, file_path: str, repo_name: str):
        fixed_repo_name = (
            self.context.autocorrect_repo_name(repo_name)
            if isinstance(self.context, AutofixContext)
            else repo_name
        )
        if not fixed_repo_name:
            return self._make_repo_not_found_error_message(repo_name)
        repo_name = fixed_repo_name

        valid_file_path = self._attempt_fix_path(file_path, repo_name)
        if valid_file_path is None:
            other_paths = self._get_potential_abs_paths(file_path, repo_name)
            return f"Error: The file path `{file_path}` doesn't exist in `{repo_name}`.\n{other_paths}".strip()

        # At this point we have ensured the file path and the repo name are valid.
        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        self.context.event_manager.add_log(f"Looking at `{file_path}` in `{repo_name}`...")

        if file_contents:
            return file_contents

        self._ensure_repos_downloaded(repo_name)

        local_read_error = None
        if repo_name in self.repo_managers:
            if not self.repo_managers[repo_name].is_available:
                return self._make_repo_unavailable_error_message(repo_name)

            repo_dir = self.repo_managers[repo_name].repo_path

            # try reading the actual file from file system
            file_contents, local_read_error = read_file_contents(repo_dir, file_path)
            if file_contents:
                return file_contents

        return f"Error: Could not read the file at path `{file_path}`.\n{local_read_error if local_read_error else ''}".strip()

    @observe(name="View Diff")
    @sentry_sdk.trace
    def view_diff(self, file_path: str, repo_name: str, commit_sha: str):
        """
        Given a file path, repository name, and commit SHA, returns the diff for the file in the given commit.
        """
        if not isinstance(self.context, AutofixContext):
            return None
        self.context.event_manager.add_log(
            f"Studying commit `{commit_sha}` in `{file_path}` in `{repo_name}`..."
        )
        patch = self.context.get_commit_patch_for_file(
            path=file_path, repo_name=repo_name, commit_sha=commit_sha
        )
        if patch is None:
            return "Could not find the file in the given commit. Either your hash is incorrect or the file does not exist in the given commit."
        return patch

    @observe(name="Explain File")
    @sentry_sdk.trace
    def explain_file(self, file_path: str, repo_name: str):
        """
        Given a file path and repository name, returns recent commits and related files.
        """
        if not isinstance(self.context, AutofixContext):
            return None
        num_commits = 30
        commit_history = self.context.get_commit_history_for_file(
            file_path, repo_name, max_commits=num_commits
        )
        if commit_history:
            return "COMMIT HISTORY:\n" + "\n".join(commit_history)
        return "No commit history found for the given file. Either the file path or repo name is incorrect, or it is just unavailable right now."

    @observe(name="Tree")
    @sentry_sdk.trace
    def tree(self, path: str, repo_name: str | None = None) -> str:
        """
        Given the path for a directory in this codebase, returns a tree representation of the directory structure and files.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name, type=self.repo_client_type)
        all_paths = repo_client.get_valid_file_paths()
        normalized_path = self._normalize_path(path)

        # Filter paths to include all files under the specified path
        files_under_path = [
            {"path": p, "status": ""}
            for p in all_paths
            if p.startswith(normalized_path) and p != normalized_path
        ]

        if not files_under_path:
            # show potential corrected paths if nothing was found here
            other_paths = self._get_potential_abs_paths(path, repo_name)
            return f"<no entries found in directory '{path or '/'}'/>\n{other_paths}".strip()

        # If more than MAX_FILES_IN_TREE files under a directory, show only the top level paths
        is_truncated = len(files_under_path) > MAX_FILES_IN_TREE

        self.context.event_manager.add_log(
            f"Viewing directory tree for `{path}` in `{repo_name}`..."
        )

        max_files_in_tree_note = (
            f"Notice: There are a total of {len(files_under_path)} files in the tree under the {normalized_path} path. Only showing immediate children, provide a more specific path to view a full tree.\n"
            if is_truncated
            else ""
        )

        # Use the _build_file_tree_string method from the repo client
        tree_representation = repo_client._build_file_tree_string(
            files_under_path,
            only_immediate_children_of_path=normalized_path if is_truncated else None,
        )
        return f"<directory_tree>\n{max_files_in_tree_note}{tree_representation}\n</directory_tree>"

    def _get_potential_abs_paths(self, path: str, repo_name: str | None = None) -> str:
        """
        Gets possible full paths for a given path.
        For example, example/path/ might actually be located at src/example/path/
        This is useful in the case that the model is using an incomplete path.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name, type=self.repo_client_type)
        all_paths = repo_client.get_valid_file_paths()
        normalized_path = self._normalize_path(path)

        # Filter paths to include parents + remove duplicates and sort
        unique_parents = sorted(
            set(
                p.split(normalized_path)[0] + normalized_path
                for p in all_paths
                if normalized_path in p and p != normalized_path
            )
        )

        if not unique_parents:
            return ""

        joined = "\n".join(unique_parents)
        return f"<did you mean>\n{joined}\n</did you mean>"

    def _attempt_fix_path(self, path: str, repo_name: str) -> str | None:
        """
        Attempts to fix a path by checking if it exists in the repository as a path or directory.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name, type=self.repo_client_type)
        all_files = repo_client.get_valid_file_paths()

        normalized_path = path.lstrip("./").lstrip("/")
        if not normalized_path:
            return None

        for p in all_files:
            if p.endswith(normalized_path):
                # is a valid file path
                return p
            if p.startswith(normalized_path):
                # is a valid directory path
                return normalized_path

        return None

    def _normalize_path(self, path: str) -> str:
        """
        Ensures paths don't start with a slash, but do end in one, such as example/path/
        """
        normalized_path = path.strip("/") + "/" if path.strip("/") else ""
        return normalized_path

    @sentry_sdk.trace
    def cleanup(self):
        """Clean up all repository clients."""
        for repo_name, local_client in list(self.repo_managers.items()):
            try:
                local_client.cleanup()
            except Exception as e:
                logger.exception(f"Error cleaning up repo {repo_name}: {e}")

        self.repo_managers = {}

    @observe(name="Search Google")
    @sentry_sdk.trace
    @inject
    def google_search(self, question: str, llm_client: LlmClient = injected):
        """
        Searches Google to answer a question.
        """
        self.context.event_manager.add_log(f'Googling "{question}"...')
        return llm_client.generate_text_from_web_search(
            prompt=question, model=GeminiProvider(model_name="gemini-2.0-flash-001")
        )

    @observe(name="Get Profile")
    @sentry_sdk.trace
    @inject
    def get_profile(self, event_id: str, rpc_client: RpcClient = injected):
        """
        Fetches a profile for a specific transaction event.
        """
        # get full event id and TraceEvent payload
        state = self.context.state.get()
        if not isinstance(self.context, AutofixContext) or not isinstance(
            state.request, AutofixRequest
        ):
            return "No trace available. Cannot fetch profiles."

        trace_tree = state.request.trace_tree
        if trace_tree is None:
            return "No trace available. Cannot fetch profiles."
        event = trace_tree.get_event_by_id(event_id)
        if event is None:
            return "Invalid event ID."
        if event.profile_id is None:
            return "No profile available for this transaction."

        self.context.event_manager.add_log(f"Studying profile for `{event.title}`...")

        # if profile available, fetch it via Seer RPC
        profile_data = rpc_client.call(
            "get_profile_details",
            organization_id=trace_tree.org_id,
            project_id=event.project_id,
            profile_id=event.profile_id,
        )  # expecting data compatible with Profile model
        if not profile_data:
            return "Could not fetch profile."
        try:
            profile = Profile.model_validate(profile_data)
            return profile.format_profile(context_before=100, context_after=100)
        except Exception as e:
            logger.exception(f"Could not parse profile from tool call: {e}")
            return "Could not fetch profile."

    @observe(name="Get Trace Event Details")
    @sentry_sdk.trace
    @inject
    def get_trace_event_details(self, event_id: str, rpc_client: RpcClient = injected):
        """
        Fetches the spans under a selected transaction event or the stacktrace under an error event.
        """
        # get full event id and TraceEvent payload
        state = self.context.state.get()
        if not isinstance(self.context, AutofixContext) or not isinstance(
            state.request, AutofixRequest
        ):
            return "No trace available. Cannot fetch details."

        trace_tree = state.request.trace_tree
        if trace_tree is None:
            return "No trace available. Cannot fetch details."
        event = trace_tree.get_event_by_id(event_id)
        if event is None:
            return "Invalid event ID."
        full_event_id = event.event_id
        if not full_event_id:
            return "Cannot fetch information for this event."

        if event.is_transaction:
            # if it's a transaction, use the spans already in the payload
            self.context.event_manager.add_log(f"Studying spans under `{event.title}`...")
            return event.format_spans_tree()
        elif event.is_error:
            # if it's an error, fetch the event details via Seer RPC
            self.context.event_manager.add_log(f"Studying connected error `{event.title}`...")

            project_id = event.project_id
            error_event_id = event.event_id
            error_data = rpc_client.call(
                "get_error_event_details",
                project_id=project_id,
                event_id=error_event_id,
            )  # expecting data compatible with SentryEventData model
            if not error_data:
                return "Could not fetch error event details."
            data = cast(SentryEventData, error_data)
            try:
                error_event_details = EventDetails.from_event(data)
                self.context.process_event_paths(error_event_details)
                return error_event_details.format_event_without_breadcrumbs()
            except Exception as e:
                logger.exception(f"Could not parse error event details from tool call: {e}")
                return "Could not fetch error event details."
        else:
            return "Cannot fetch information for this event."

    @observe(name="Ripgrep Search")
    @sentry_sdk.trace
    def run_ripgrep(
        self,
        *,
        query: str,
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
        case_sensitive: bool = False,
        repo_name: str | None = None,
        use_regex: bool = False,
    ) -> str:
        self._ensure_repos_downloaded(repo_name)

        if not query:
            return "Error: query is required for ripgrep search"

        cmd = ["rg", f'"{query}"']

        # We wanna ignore long lines
        cmd.extend(["--max-columns", "1024"])

        # limit threads
        cmd.extend(["--threads", "2"])

        if not use_regex:
            cmd.append("--fixed-strings")

        if not case_sensitive:
            cmd.append("--ignore-case")

        if include_pattern:
            cmd.extend(["--glob", f'"{include_pattern}"'])

        if exclude_pattern:
            cmd.extend(["--glob", f'"!{exclude_pattern}"'])

        if repo_name:
            # Single repository search
            if repo_name not in self.repo_managers:
                return f"Error: Repository {repo_name} not found."
            if not self.repo_managers[repo_name].is_available:
                return self._make_repo_unavailable_error_message(repo_name)

            tmp_repo_dir = self.repo_managers[repo_name].repo_path

            cmd.append(tmp_repo_dir)

            return run_ripgrep_in_repo(tmp_repo_dir, cmd)
        else:
            # Multiple repository search - we'll need to run separate commands for each repo
            # and combine results
            repo_names = self._get_repo_names()

            # Run ripgrep in parallel across repositories
            def search_repo(repo_name: str) -> tuple[str, str] | None:
                if repo_name not in self.repo_managers:
                    return (repo_name, f"Error: Repository {repo_name} not found or not downloaded")
                if not self.repo_managers[repo_name].is_available:
                    return (repo_name, self._make_repo_unavailable_error_message(repo_name))

                repo_dir = self.repo_managers[repo_name].repo_path
                try:
                    result = run_ripgrep_in_repo(repo_dir, cmd)
                    return (repo_name, result)
                except Exception as e:
                    logger.exception(f"Error searching repo {repo_name}: {e}")
                    return (repo_name, f"Error: {e}")

            results = []
            with ThreadPoolExecutor(initializer=copy_modules_initializer()) as executor:
                futures = {
                    executor.submit(search_repo, repo_name): repo_name for repo_name in repo_names
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

            return "\n".join(f"Result for {rn}:\n{result}" for rn, result in results)

    @observe(name="Find Files")
    @sentry_sdk.trace
    def find_files(self, command: str, repo_name: str | None = None):
        """
        Runs a `find` command over the downloaded repositories to search for files.
        """
        if not command.startswith("find "):
            return "Command must be a valid find command that starts with 'find'."

        command = command.replace('\\"', '"')  # un-escape escaped quotes
        command = command.replace("\\'", "'")  # un-escape escaped single quotes
        command = command.replace("\\\\", "\\")  # un-escape escaped backslashes

        self.context.event_manager.add_log(f"Searching files with `{command}`...")

        self._ensure_repos_downloaded(repo_name)

        repo_names = [repo_name] if repo_name else self._get_repo_names()
        all_results = []

        # Parse the command into a list of arguments
        try:
            cmd_args = shlex.split(command)
        except Exception as e:
            return f"Error parsing find command: {str(e)}"

        for repo_name in repo_names:
            if repo_name not in self.repo_managers:
                continue
            if not self.repo_managers[repo_name].is_available:
                all_results.append(self._make_repo_unavailable_error_message(repo_name))
                continue

            tmp_repo_dir = self.repo_managers[repo_name].repo_path
            if not tmp_repo_dir:
                continue

            try:
                # Run the find command in the repo directory
                try:
                    process = subprocess.run(
                        cmd_args,
                        shell=False,
                        cwd=tmp_repo_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=45,
                    )

                    if process.returncode != 0:
                        all_results.append(f"Results from {repo_name}: {process.stderr}")
                    elif process.stdout:
                        all_results.append(
                            f"Results from {repo_name}:\n------\n{process.stdout}\n------"
                        )
                    else:
                        all_results.append(f"Results from {repo_name}: no files found.")
                except subprocess.TimeoutExpired:
                    all_results.append(
                        f"Results from {repo_name}: command timed out. Try narrowing your search."
                    )
            except Exception as e:
                all_results.append(f"Error in repo {repo_name}: {str(e)}")

        if not all_results:
            return "No results found."

        return "\n\n".join(all_results)

    def _append_file_change(self, repo_name: str, file_change: FileChange):
        with self.context.state.update() as cur:
            for repo in cur.request.repos:
                if repo.full_name == repo_name:
                    cur.codebases[repo.external_id].file_changes.append(file_change)

                    return True

        return False

    def _get_repo_name_and_path(
        self, kwargs: dict[str, Any], allow_nonexistent_paths: bool = False
    ) -> tuple[str | None, str | None, str | None]:
        repos = self._get_repo_names()

        path_args = kwargs.get("path", None)
        repo_name = None

        if not repos:
            return "Error: No repositories found.", None, None

        path: str
        if len(repos) > 1:
            if ":" not in path_args:
                return (
                    "Error: Multiple repositories found. Please provide a repository name in the format `repo_name:path`, such as `repo_owner/repo:src/foo/bar.py`. The repositories available to you are: "
                    + ", ".join(repos),
                    None,
                    None,
                )
            segments = path_args.split(":")
            repo_name = segments[0]
            path = segments[1]
        else:
            repo_name = repos[0]
            path = path_args

        fixed_path = self._attempt_fix_path(path, repo_name)
        if not fixed_path:
            if allow_nonexistent_paths:
                return None, repo_name, path

            return (
                f"Error: The path you provided '{path}' does not exist in the repository '{repo_name}'.",
                None,
                None,
            )

        return None, repo_name, fixed_path

    @observe(name="Claude Tools")
    @sentry_sdk.trace
    def handle_claude_tools(self, **kwargs: Any) -> str:
        """
        Handles various file editing commands from Claude tools.

        Args:
            **kwargs: Dictionary containing:
                - command: The type of command to execute ("view", "str_replace", "create", "insert", "undo_edit")
                - path: The file path to operate on
                - repo_name: The repository name (optional if only one repo)
                - Additional command-specific parameters

        Returns:
            str: Success message or error description
        """
        command = kwargs.get("command", "")
        error, repo_name, path = self._get_repo_name_and_path(
            kwargs, allow_nonexistent_paths=command in ["create", "undo_edit"]
        )

        if error:
            return error

        if not path:
            return "Error: Path could not be resolved"

        if not repo_name:
            return "Error: Repo could not be resolved"

        tool_call_id = kwargs.get("tool_call_id", None)
        current_memory_index = kwargs.get("current_memory_index", -1)

        fixed_repo_name = (
            self.context.autocorrect_repo_name(repo_name)
            if isinstance(self.context, AutofixContext)
            else repo_name
        )
        if not fixed_repo_name:
            return self._make_repo_not_found_error_message(repo_name)
        repo_name = fixed_repo_name

        command_handlers = {
            "view": self._handle_view_command,
            "str_replace": self._handle_str_replace_command,
            "create": self._handle_create_command,
            "insert": self._handle_insert_command,
            "undo_edit": self._handle_undo_edit_command,
        }

        handler = command_handlers.get(command)
        if handler:
            return handler(
                kwargs,
                repo_name,
                path,
                tool_call_id=tool_call_id,
                current_memory_index=current_memory_index,
            )

        return f"Error: Unknown command '{command}'"

    def _get_file_contents(self, path: str, repo_name: str) -> str:
        """Helper method to get file contents with proper error handling."""
        contents = self.context.get_file_contents(path, repo_name=repo_name)
        if not contents:
            raise ValueError("File not found")
        return contents

    def _create_file_change(
        self,
        change_type: str,
        reference_snippet: str,
        new_snippet: str,
        path: str,
        repo_name: str,
        commit_message: str = "COMMIT",
        tool_call_id: str | None = None,
    ) -> FileChange:
        """Helper method to create a FileChange instance."""
        return FileChange(
            change_type=change_type,
            commit_message=commit_message,
            reference_snippet=reference_snippet,
            new_snippet=new_snippet,
            path=path,
            repo_name=repo_name,
            tool_call_id=tool_call_id,
        )

    def _apply_file_change(self, repo_name: str, file_change: FileChange) -> str:
        """Helper method to apply a file change with proper error handling."""
        try:
            if self._append_file_change(repo_name, file_change):
                return "Change applied successfully."
            return "Error: Failed to apply file change."
        except Exception as e:
            return f"Error: Failed to apply changes to file: {str(e)}"

    @observe(name="View")
    @sentry_sdk.trace
    def _handle_view_command(
        self, kwargs: dict[str, Any], repo_name: str, path: str, **extra_kwargs: Any
    ) -> str:
        """Handles the view command to display file contents with optional line range."""
        try:
            view_range = kwargs.get("view_range", [])

            # handle directories
            if os.path.isdir(path):
                if view_range:
                    return "Error: Cannot view a directory with a line range."

                return self.tree(path, repo_name)

            file_contents = self._get_file_contents(path, repo_name)
            lines = file_contents.split("\n")

            if view_range:
                try:
                    start_line = max(0, int(view_range[0]) - 1)
                    end_line = min(len(lines), int(view_range[1]))
                    if start_line >= end_line:
                        return "Error: Invalid line range - start must be less than end"
                    lines = lines[start_line:end_line]
                    self.context.event_manager.add_log(
                        f"Looking at lines `{start_line+1}` to `{end_line}` of `{path}`..."
                    )
                except (ValueError, IndexError):
                    return "Error: Invalid line range format"
            else:
                self.context.event_manager.add_log(f"Looking at `{path}`...")

            return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
        except ValueError as e:
            return str(e)

    @observe(name="String Replace")
    @sentry_sdk.trace
    def _handle_str_replace_command(
        self,
        kwargs: dict[str, Any],
        repo_name: str,
        path: str,
        tool_call_id: str | None = None,
        current_memory_index: int = -1,
    ) -> str:
        """Handles the string replace command to replace text in a file."""
        old_str = kwargs.get("old_str")
        new_str = kwargs.get("new_str")
        if not old_str or not new_str:
            return "Error: old_str and new_str are required for str_replace command"

        try:
            file_change = self._create_file_change(
                "edit", old_str, new_str, path, repo_name, tool_call_id=tool_call_id
            )

            self.context.event_manager.add_log(f"Making an edit to `{path}` in `{repo_name}`...")

            document = BaseDocument(
                path=path,
                repo_name=repo_name,
                text=self.context.get_file_contents(path, repo_name),
            )

            file_diff, _ = make_file_patches([file_change], [path], [document])

            if not file_diff:
                return "Error: No changes were made to the file. Make sure old_str is exact, even including indentation."

            self.context.event_manager.send_insight(
                InsightSharingOutput(
                    insight=f"Edited `{path}` in `{repo_name}`.",
                    change_diff=file_diff,
                    generated_at_memory_index=current_memory_index,
                    type=InsightSharingType.FILE_CHANGE,
                )
            )

            return self._apply_file_change(repo_name, file_change)
        except ValueError as e:
            return str(e)

    @observe(name="Create File")
    @sentry_sdk.trace
    def _handle_create_command(
        self,
        kwargs: dict[str, Any],
        repo_name: str,
        path: str,
        tool_call_id: str | None = None,
        current_memory_index: int = -1,
    ) -> str:
        """Handles the create command to create a new file."""
        file_text = kwargs.get("file_text", "")
        if not file_text:
            return "Error: file_text is required for create command"

        existing_content = self.context.get_file_contents(path, repo_name=repo_name)
        if existing_content is not None:
            return f"Error: Cannot create file '{path}' because it already exists."

        file_change = self._create_file_change(
            "create", file_text, file_text, path, repo_name, tool_call_id=tool_call_id
        )

        document = BaseDocument(
            path=path,
            repo_name=repo_name,
            text="",
        )

        file_diff, _ = make_file_patches([file_change], [path], [document])

        if not file_diff:
            return "Error: No changes were made to the file."

        self.context.event_manager.add_log(f"Creating a new file `{path}` in `{repo_name}`...")
        self.context.event_manager.send_insight(
            InsightSharingOutput(
                insight=f"Created file `{path}` in `{repo_name}`.",
                change_diff=file_diff,
                generated_at_memory_index=current_memory_index,
                type=InsightSharingType.FILE_CHANGE,
            )
        )

        return self._apply_file_change(repo_name, file_change)

    @observe(name="Insert Text")
    @sentry_sdk.trace
    def _handle_insert_command(
        self,
        kwargs: dict[str, Any],
        repo_name: str,
        path: str,
        tool_call_id: str | None = None,
        current_memory_index: int = -1,
    ) -> str:
        """Handles the insert command to insert text at a specific line."""
        try:
            insert_line = kwargs.get("insert_line")
            if insert_line is None:
                return "Error: insert_line is required for insert command"

            insert_line = int(insert_line)
            new_str = kwargs.get("insert_text", "")
            if not new_str:
                return "Error: insert_text is required for insert command"

            file_contents = self._get_file_contents(path, repo_name)
            lines = file_contents.split("\n")
            if not 0 <= insert_line <= len(lines):
                return f"Error: Invalid line number. Must be between 0 and {len(lines)}"

            lines.insert(insert_line, new_str)
            new_file_contents = "\n".join(lines)
            file_change = self._create_file_change(
                "edit", file_contents, new_file_contents, path, repo_name, tool_call_id=tool_call_id
            )

            document = BaseDocument(
                path=path,
                repo_name=repo_name,
                text=self.context.get_file_contents(path, repo_name),
            )

            file_diff, _ = make_file_patches([file_change], [path], [document])

            if not file_diff:
                return "Error: No changes were made to the file."

            self.context.event_manager.add_log(f"Making a change to `{path}` in `{repo_name}`...")
            self.context.event_manager.send_insight(
                InsightSharingOutput(
                    insight=f"Edited `{path}` in `{repo_name}`.",
                    change_diff=file_diff,
                    generated_at_memory_index=current_memory_index,
                    type=InsightSharingType.FILE_CHANGE,
                )
            )

            return self._apply_file_change(repo_name, file_change)
        except ValueError as e:
            return str(e)

    @observe(name="Undo Edit")
    @sentry_sdk.trace
    def _handle_undo_edit_command(
        self, kwargs: dict[str, Any], repo_name: str, path: str, **extra_kwargs: Any
    ) -> str:
        """Handles the undo edit command to remove file changes."""
        with self.context.state.update() as cur:
            for repo in cur.request.repos:
                if repo.full_name == repo_name:
                    codebase = cur.codebases[repo.external_id]
                    if not codebase:
                        return "Error: No codebases found"

                    # Remove all file changes for this path
                    codebase.file_changes = [fc for fc in codebase.file_changes if fc.path != path]

                    self.context.event_manager.add_log(
                        f"Undoing edits to `{path}` in `{repo_name}`..."
                    )

                    return "File changes undone successfully."
            return "Error: No file changes found to undo."

    def get_tools(
        self, can_access_repos: bool = True, include_claude_tools: bool = False
    ) -> list[ClaudeTool | FunctionTool]:
        tools: list[ClaudeTool | FunctionTool] = [
            FunctionTool(
                name="google_search",
                fn=self.google_search,
                description="Searches the web with Google and returns the answer to a question.",
                parameters=[
                    {
                        "name": "question",
                        "type": "string",
                        "description": "The question you want to answer.",
                    },
                ],
                required=["question"],
            )
        ]

        if include_claude_tools:
            tools.extend(
                [
                    ClaudeTool(
                        type="text_editor_20250124",
                        name="str_replace_editor",
                        fn=self.handle_claude_tools,
                    )
                ]
            )

        if can_access_repos:
            tools.extend(
                [
                    FunctionTool(
                        name="tree",
                        fn=self.tree,
                        description="Given the path for a directory in this codebase, returns a tree representation of the directory structure and files.",
                        parameters=[
                            {
                                "name": "path",
                                "type": "string",
                                "description": 'The path to view. For example, "src/app/components"',
                            },
                            {
                                "name": "repo_name",
                                "type": "string",
                                "description": "Optional name of the repository to search in if you know it.",
                            },
                        ],
                        required=["path"],
                    ),
                    FunctionTool(
                        name="expand_document",
                        fn=self.expand_document,
                        description=textwrap.dedent(
                            """\
                    Given a document path, returns the entire document text.
                    - Note: To save time and money, if you're looking to expand multiple documents, call this tool multiple times in the same message.
                    - If a document has already been expanded earlier in the conversation, don't use this tool again for the same file path."""
                        ),
                        parameters=[
                            {
                                "name": "file_path",
                                "type": "string",
                                "description": "The document path to expand.",
                            },
                            {
                                "name": "repo_name",
                                "type": "string",
                                "description": "Name of the repository containing the file.",
                            },
                        ],
                        required=["file_path", "repo_name"],
                    ),
                    FunctionTool(
                        name="grep_search",
                        fn=self.run_ripgrep,
                        description="Runs a ripgrep command over the codebase to find what you're looking for. Use this as your main tool for searching codebases. Use the include and exclude patterns to narrow down the search to specific paths or file types.",
                        parameters=[
                            {
                                "name": "include_pattern",
                                "type": "string",
                                "description": "Optional glob pattern for files to include. For example, '*.py' for Python files.",
                            },
                            {
                                "name": "exclude_pattern",
                                "type": "string",
                                "description": "Optional glob pattern for files to exclude. For example, '*.test.py' for test files.",
                            },
                            {
                                "name": "case_sensitive",
                                "type": "boolean",
                                "description": "Whether the search should be case sensitive.",
                            },
                            {
                                "name": "use_regex",
                                "type": "boolean",
                                "description": "Set this to true to search for a regex pattern. By default set to false.",
                            },
                            {
                                "name": "repo_name",
                                "type": "string",
                                "description": "Optional name of the repository to search in. If not provided, all repositories will be searched.",
                            },
                            {
                                "name": "query",
                                "type": "string",
                                "description": "The precise query you're searching for. By default interpreted as a literal string, so no escaping is needed. Set use_regex=true to use regex pattern matching.",
                            },
                        ],
                        required=["query"],
                    ),
                    FunctionTool(
                        name="find_files",
                        fn=self.find_files,
                        description="Runs a find command over the codebase to search for files based on various criteria.",
                        parameters=[
                            {
                                "name": "command",
                                "type": "string",
                                "description": "The full find command to execute. Do NOT include repo names in your command. Example: 'find . -name \"*.py\" -type f'",
                            },
                            {
                                "name": "repo_name",
                                "type": "string",
                                "description": "Optional name of the repository to search in. If not provided, all repositories will be searched.",
                            },
                        ],
                        required=["command"],
                    ),
                    FunctionTool(
                        name="semantic_file_search",
                        fn=self.semantic_file_search,
                        description="Tries to find the file in the codebase that contains what you're looking for.",
                        parameters=[
                            {
                                "name": "query",
                                "type": "string",
                                "description": "Describe what file you're looking for.",
                            },
                        ],
                        required=["query"],
                    ),
                ]
            )

        if isinstance(self.context, AutofixContext):
            tools.extend(
                [
                    FunctionTool(
                        name="explain_file",
                        fn=self.explain_file,
                        description="Given a file path and repository name, describes recent commits and suggests related files.",
                        parameters=[
                            {
                                "name": "file_path",
                                "type": "string",
                                "description": "The file to get more context on.",
                            },
                            {
                                "name": "repo_name",
                                "type": "string",
                                "description": "Name of the repository containing the file.",
                            },
                        ],
                        required=["file_path", "repo_name"],
                    ),
                    FunctionTool(
                        name="view_diff",
                        fn=self.view_diff,
                        description="Given a file path, repository name, and 7 character commit SHA, returns the diff of what changed in the file in the given commit.",
                        parameters=[
                            {
                                "name": "file_path",
                                "type": "string",
                                "description": "The file path to view the diff for.",
                            },
                            {
                                "name": "repo_name",
                                "type": "string",
                                "description": "Name of the repository containing the file.",
                            },
                            {
                                "name": "commit_sha",
                                "type": "string",
                                "description": "The 7 character commit SHA to view the diff for.",
                            },
                        ],
                        required=["file_path", "repo_name", "commit_sha"],
                    ),
                ]
            )

        # if (
        #     isinstance(self.context, AutofixContext)
        #     and not self.context.state.get().request.options.disable_interactivity
        # ):
        #     tools.append(
        #         FunctionTool(
        #             name="ask_a_question",
        #             fn=self.ask_user_question,
        #             description="Ask the user a question about business logic, product requirements, past decisions, or subjective preferences. You may not ask about anything else. Only use this tool if necessary.",
        #             parameters=[
        #                 {
        #                     "name": "question",
        #                     "type": "string",
        #                     "description": "The question you want to ask.",
        #                 }
        #             ],
        #             required=["question"],
        #         )
        #     )

        run_request = self.context.state.get().request
        if (
            isinstance(self.context, AutofixContext)
            and isinstance(run_request, AutofixRequest)
            and not run_request.options.disable_interactivity
        ):
            trace_tree = run_request.trace_tree

            if (
                trace_tree and trace_tree.events
            ):  # Only add trace events tool if there are events in the trace
                tools.append(
                    FunctionTool(
                        name="get_trace_event_details",
                        fn=self.get_trace_event_details,
                        description="Read detailed information about a specific event in the trace. You can view stack traces for connected errors and view the granular spans that make up other events, like endpoint calls, background jobs, and page loads.",
                        parameters=[
                            {
                                "name": "event_id",
                                "type": "string",
                                "description": "The ID of the transaction event to fetch the details for.",
                            },
                        ],
                        required=["event_id"],
                    )
                )

            if trace_tree and any(
                event.profile_id is not None for event in trace_tree._get_all_events()
            ):  # Only add profile tool if there are events with profile_ids in the trace
                tools.append(
                    FunctionTool(
                        name="get_profile_for_trace_event",
                        fn=self.get_profile,
                        description="Read a record of the exact code execution for a specific event in the trace (any event marked with 'profile available').",
                        parameters=[
                            {
                                "name": "event_id",
                                "type": "string",
                                "description": "The event ID of the event to fetch the profile for.",
                            },
                        ],
                        required=["event_id"],
                    )
                )

        return tools


class SemanticSearchTools(BaseTools):
    def get_tools(
        self, can_access_repos: bool = True, include_claude_tools: bool = False
    ) -> list[ClaudeTool | FunctionTool]:
        if not can_access_repos:
            return []

        return [
            FunctionTool(
                name="tree",
                fn=self.tree,
                description="Given the path for a directory in this codebase, returns a tree representation of the directory structure and files.",
                parameters=[
                    {
                        "name": "path",
                        "type": "string",
                        "description": 'The path to view. For example, "src/app/components"',
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "Optional name of the repository to search in if you know it.",
                    },
                ],
                required=["path"],
            ),
            FunctionTool(
                name="expand_document",
                fn=self.expand_document,
                description=textwrap.dedent(
                    """\
                Given a document path, returns the entire document text.
                - Note: To save time and money, if you're looking to expand multiple documents, call this tool multiple times in the same message.
                - If a document has already been expanded earlier in the conversation, don't use this tool again for the same file path."""
                ),
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The document path to expand.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "Name of the repository containing the file.",
                    },
                ],
                required=["file_path", "repo_name"],
            ),
        ]
