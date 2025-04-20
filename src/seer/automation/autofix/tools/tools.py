import logging
import os
import shlex
import subprocess
import textwrap
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed
from threading import Lock
from typing import Any, cast

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.tools import ClaudeTool, FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import (
    InsightSharingOutput,
    InsightSharingType,
)
from seer.automation.autofix.models import AutofixRequest
from seer.automation.codebase.file_patches import make_file_patches
from seer.automation.codebase.models import BaseDocument
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codebase.utils import cleanup_dir
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.models import EventDetails, FileChange, Profile, SentryEventData
from seer.dependency_injection import copy_modules_initializer, inject, injected
from seer.langfuse import append_langfuse_observation_metadata
from seer.rpc import RpcClient

from .grep_search import run_grep_search

logger = logging.getLogger(__name__)

MAX_FILES_IN_TREE = 100
GREP_TIMEOUT_SECONDS = 10
MAX_GREP_LINE_CHARACTER_LENGTH = 1000
TOTAL_GREP_RESULTS_CHARACTER_LENGTH = 20000


class BaseTools:
    context: AutofixContext | CodegenContext
    retrieval_top_k: int
    tmp_dir: dict[str, tuple[str, str]] = {}  # Maps repo_name to (tmp_dir, tmp_repo_dir)
    tmp_repo_dir: str | None = None
    repo_client_type: RepoClientType = RepoClientType.READ
    _download_future: Future | None = None

    def __init__(
        self,
        context: AutofixContext | CodegenContext,
        retrieval_top_k: int = 8,
        repo_client_type: RepoClientType = RepoClientType.READ,
    ):
        self.context = context
        self.retrieval_top_k = retrieval_top_k
        self.repo_client_type = repo_client_type
        self.tmp_dir = {}

        # Start downloading repos in parallel immediately
        self._executor = ThreadPoolExecutor(initializer=copy_modules_initializer())
        self._start_parallel_repo_download()

    def _start_parallel_repo_download(self):
        """Start downloading all repositories in parallel in the background."""
        repo_names = self._get_repo_names()
        if not repo_names:
            return

        def download_all_repos():
            """Download all repositories and update self.tmp_dir directly."""
            # Create a lock for thread-safe updates to self.tmp_dir
            if not hasattr(self, "_tmp_dir_lock"):
                self._tmp_dir_lock = Lock()

            for repo_name in repo_names:
                # Skip if already downloaded or in progress
                with self._tmp_dir_lock:
                    if repo_name in self.tmp_dir:
                        continue

                try:
                    repo_client = self.context.get_repo_client(
                        repo_name=repo_name, type=self.repo_client_type
                    )
                    tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()

                    # Update self.tmp_dir in a thread-safe way
                    with self._tmp_dir_lock:
                        self.tmp_dir[repo_name] = (tmp_dir, tmp_repo_dir)

                except Exception as e:
                    logger.exception(f"Error pre-downloading repo {repo_name}: {e}")

            return True  # Signal completion

        self._download_future = self._executor.submit(download_all_repos)

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

    @observe(name="Semantic File Search")
    @ai_track(description="Semantic File Search")
    @inject
    def semantic_file_search(self, query: str, llm_client: LlmClient = injected):
        from seer.automation.autofix.tools.semantic_search import semantic_search

        self.context.event_manager.add_log(f'Searching for "{query}"...')

        result = semantic_search(query=query, context=self.context)

        if not result:
            return "Could not figure out which file matches what you were looking for. You'll have to try yourself."

        return result

    @observe(name="Expand Document")
    @ai_track(description="Expand Document")
    def expand_document(self, file_path: str, repo_name: str):
        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        self.context.event_manager.add_log(f"Looking at `{file_path}` in `{repo_name}`...")

        if file_contents:
            return file_contents

        # show potential corrected paths if nothing was found here
        other_paths = self._get_potential_abs_paths(file_path, repo_name)
        return f"<document with the provided path not found/>\n{other_paths}".strip()

    @observe(name="View Diff")
    @ai_track(description="View Diff")
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
    @ai_track(description="Explain File")
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
    @ai_track(description="Tree")
    def tree(self, path: str, repo_name: str | None = None) -> str:
        """
        Given the path for a directory in this codebase, returns a tree representation of the directory structure and files.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name, type=self.repo_client_type)
        all_paths = repo_client.get_index_file_set()
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
        all_paths = repo_client.get_index_file_set()
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
        all_files = repo_client.get_index_file_set()

        for p in all_files:
            if p.endswith(path):
                # is a valid file path
                return p
            if p.startswith(path):
                # is a valid directory path
                return path

        return None

    def _normalize_path(self, path: str) -> str:
        """
        Ensures paths don't start with a slash, but do end in one, such as example/path/
        """
        normalized_path = path.strip("/") + "/" if path.strip("/") else ""
        return normalized_path

    def cleanup(self):
        # Clean up any in-progress downloads
        if self._download_future and not self._download_future.done():
            try:
                self._download_future.cancel()
            except Exception as e:
                logger.exception(f"Error cancelling downloads during cleanup: {e}")
            self._download_future = None

        # Clean up any tmp dirs that were created
        if self.tmp_dir:
            for tmp_dir, _ in self.tmp_dir.values():
                cleanup_dir(tmp_dir)
        # Reset tmp_dir
        self.tmp_dir = {}

    @observe(name="Search Google")
    @ai_track(description="Search Google")
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
    @ai_track(description="Get Profile")
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
    @ai_track(description="Get Trace Event Details")
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

    @observe(name="Grep Search")
    @ai_track(description="Grep Search")
    def grep_search(self, command: str, repo_name: str | None = None):
        """
        Runs a grep command over the downloaded repositories.
        """
        if not command.startswith("grep "):
            return "Command must be a valid grep command that starts with 'grep'."

        command = command.replace('\\"', '"')  # un-escape escaped quotes
        command = command.replace("\\'", "'")  # un-escape escaped single quotes
        command = command.replace("\\\\", "\\")  # un-escape escaped backslashes

        self.context.event_manager.add_log(f"Grepping codebase with `{command}`...")

        self._ensure_repos_downloaded(repo_name)

        repo_names = [repo_name] if repo_name else self._get_repo_names()

        # Parse the command into a list of arguments
        try:
            cmd_args = shlex.split(command)
        except Exception as e:
            return f"Error parsing grep command: {str(e)}"

        return run_grep_search(cmd_args, repo_names, self.tmp_dir)

    def _ensure_repos_downloaded(self, repo_name: str | None = None):
        """
        Helper method to ensure repositories are downloaded to temporary directories.
        Checks if downloads are in progress and waits for them if necessary,
        or triggers downloads for missing repositories.

        Args:
            repo_name: If provided, only ensures this specific repo is downloaded.
                      If None, ensures all repos are downloaded.
        """
        # Check if parallel download is in progress
        if self._download_future is not None:
            if self._download_future.done():
                # Download completed - process results
                try:
                    self._download_future.result()
                except Exception as e:
                    logger.exception(f"Error in parallel repo download: {e}")
                finally:
                    self._download_future = None
            else:
                # Download in progress - wait for it to complete with a timeout
                try:
                    self._download_future.result(timeout=60)
                    self._download_future = None
                except TimeoutError:
                    logger.warning(
                        "Parallel repo download taking too long, proceeding with individual downloads"
                    )
                    self._download_future = None
                except Exception as e:
                    logger.exception(f"Error waiting for parallel repo download: {e}")
                    self._download_future = None

        if repo_name:
            repo_names_to_download = [repo_name] if repo_name not in self.tmp_dir else []
        else:
            repo_names_to_download = [rn for rn in self._get_repo_names() if rn not in self.tmp_dir]

        if not repo_names_to_download:
            return

        append_langfuse_observation_metadata({"repo_download": True})

        if not hasattr(self, "_tmp_dir_lock"):
            self._tmp_dir_lock = Lock()

        if len(repo_names_to_download) == 1:
            # Single repo - download synchronously
            try:
                repo_name = repo_names_to_download[0]
                repo_client = self.context.get_repo_client(
                    repo_name=repo_name, type=self.repo_client_type
                )
                tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()

                with self._tmp_dir_lock:
                    self.tmp_dir[repo_name] = (tmp_dir, tmp_repo_dir)
            except Exception as e:
                logger.exception(f"Error downloading repo {repo_name}: {e}")
        else:
            # Multiple repos - download in parallel
            def download_repo(repo_name):
                try:
                    repo_client = self.context.get_repo_client(
                        repo_name=repo_name, type=self.repo_client_type
                    )
                    tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()
                    return repo_name, (tmp_dir, tmp_repo_dir)
                except Exception as e:
                    logger.exception(f"Error downloading repo {repo_name}: {e}")
                    return None

            with ThreadPoolExecutor(initializer=copy_modules_initializer()) as executor:
                futures = {
                    executor.submit(download_repo, repo_name): repo_name
                    for repo_name in repo_names_to_download
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        repo_name, repo_dirs = result
                        with self._tmp_dir_lock:
                            if repo_name is not None:
                                self.tmp_dir[repo_name] = repo_dirs

    @observe(name="Find Files")
    @ai_track(description="Find Files")
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
            if repo_name not in self.tmp_dir:
                continue
            tmp_dir, tmp_repo_dir = self.tmp_dir[repo_name]
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
        self, kwargs: dict[str, Any]
    ) -> tuple[str | None, str | None, str | None]:
        repos = self._get_repo_names()

        path_args = kwargs.get("path", None)
        repo_name = None

        if not repos:
            return "Error: No repositories found.", None, None

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
            return (
                f"Error: The path you provided '{path}' does not exist in the repository '{repo_name}'.",
                None,
                None,
            )

        return None, repo_name, path

    @observe(name="Claude Tools")
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
        error, repo_name, path = self._get_repo_name_and_path(kwargs)

        if error:
            return error

        tool_call_id = kwargs.get("tool_call_id", None)
        current_memory_index = kwargs.get("current_memory_index", -1)

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
                return "Error: No changes were made to the file."

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
                        fn=self.grep_search,
                        description="Runs a grep command over the codebase to find what you're looking for.",
                        parameters=[
                            {
                                "name": "command",
                                "type": "string",
                                "description": "The full grep command to execute. Do NOT include repo names in your command. Remember, you can get more context with the -C flag.",
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
