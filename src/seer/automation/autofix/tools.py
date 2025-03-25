import logging
import shlex
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, TypeAlias, cast

from langfuse.decorators import observe
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.models import AutofixRequest
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codebase.utils import cleanup_dir
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.models import EventDetails, Profile, SentryEventData
from seer.dependency_injection import inject, injected
from seer.langfuse import append_langfuse_observation_metadata
from seer.rpc import RpcClient

logger = logging.getLogger(__name__)


class BaseTools:
    context: AutofixContext | CodegenContext
    retrieval_top_k: int
    tmp_dir: dict[str, tuple[str, str]] | None = None  # Maps repo_name to (tmp_dir, tmp_repo_dir)
    tmp_repo_dir: str | None = None
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _get_repo_names(self) -> list[str]:
        if isinstance(self.context, AutofixContext):
            return [repo.full_name for repo in self.context.state.get().readable_repos]
        elif isinstance(self.context, CodegenContext):
            return [self.context.repo.full_name]
        else:
            raise ValueError(f"Unsupported context type: {type(self.context)}")

    def _semantic_file_search_completion(
        self, query: str, valid_file_paths: str, repo_names: list[str], llm_client: LlmClient
    ):
        prompt = textwrap.dedent(
            """
            I'm searching for the file in this codebase that contains {query}. Please pick the most relevant file from the following list:
            ------------
            {valid_file_paths}
            """
        ).format(query=query, valid_file_paths=valid_file_paths)

        if 2 <= len(repo_names) < 100:
            # Lower bound avoids Gemini-Pydantic incompatibility.
            # Upper bound is b/c structured output can't handle too many options in a Literal.
            RepoName: TypeAlias = Literal[tuple(repo_names)]  # type: ignore[valid-type]
        else:
            RepoName: TypeAlias = str  # type: ignore[no-redef]

        class FileLocation(BaseModel):
            file_path: str
            repo_name: RepoName  # type: ignore

        response = llm_client.generate_structured(
            prompt=prompt,
            model=GeminiProvider(model_name="gemini-2.0-flash-001"),
            response_format=FileLocation,
        )
        return response.parsed

    @observe(name="Semantic File Search")
    @ai_track(description="Semantic File Search")
    @inject
    def semantic_file_search(self, query: str, llm_client: LlmClient = injected):
        repo_names = self._get_repo_names()
        files_per_repo = {}
        for repo_name in repo_names:
            repo_client = self.context.get_repo_client(
                repo_name=repo_name, type=self.repo_client_type
            )
            valid_file_paths = repo_client.get_valid_file_paths(files_only=True)

            # Convert the list of file paths to a tree structure
            files_with_status = [{"path": path, "status": ""} for path in valid_file_paths]
            tree_representation = repo_client._build_file_tree_string(files_with_status)
            files_per_repo[repo_name] = tree_representation

        self.context.event_manager.add_log(f'Searching for "{query}"...')

        all_valid_paths = "\n".join(
            [
                f"FILES IN REPO {repo_name}:\n{files_per_repo[repo_name]}\n------------"
                for repo_name in repo_names
            ]
        )
        file_location = self._semantic_file_search_completion(
            query, all_valid_paths, repo_names, llm_client
        )
        file_path = file_location.file_path if file_location else None
        repo_name = file_location.repo_name if file_location else None
        if file_path is None or repo_name is None:
            return "Could not figure out which file matches what you were looking for. You'll have to try yourself."

        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        if file_contents is None:
            return "Could not figure out which file matches what you were looking for. You'll have to try yourself."

        return f"This file might be what you're looking for: `{file_path}`. Contents:\n\n{file_contents}"

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

        self.context.event_manager.add_log(
            f"Viewing directory tree for `{path}` in `{repo_name}`..."
        )

        # Use the _build_file_tree_string method from the repo client
        tree_representation = repo_client._build_file_tree_string(files_under_path)
        return f"<directory_tree>\n{tree_representation}\n</directory_tree>"

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

    def _normalize_path(self, path: str) -> str:
        """
        Ensures paths don't start with a slash, but do end in one, such as example/path/
        """
        normalized_path = path.strip("/") + "/" if path.strip("/") else ""
        return normalized_path

    def cleanup(self):
        if self.tmp_dir:
            # Clean up all tmp dirs
            for tmp_dir, _ in self.tmp_dir.values():
                cleanup_dir(tmp_dir)
            self.tmp_dir = None

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
        all_results = []

        # Parse the command into a list of arguments
        try:
            cmd_args = shlex.split(command)
        except Exception as e:
            return f"Error parsing grep command: {str(e)}"

        for repo_name in repo_names:
            if self.tmp_dir is None or repo_name not in self.tmp_dir:
                continue
            tmp_dir, tmp_repo_dir = self.tmp_dir[repo_name]
            if not tmp_repo_dir:
                continue

            try:
                # Run the grep command in the repo directory
                process = subprocess.run(
                    cmd_args,
                    shell=False,
                    cwd=tmp_repo_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if (
                    process.returncode != 0 and process.returncode != 1
                ):  # grep returns 1 when no matches found
                    all_results.append(f"Results from {repo_name}: {process.stderr}")
                elif process.stdout:
                    all_results.append(
                        f"Results from {repo_name}:\n------\n{process.stdout}\n------"
                    )
                else:
                    all_results.append(f"Results from {repo_name}: no results found.")
            except Exception as e:
                all_results.append(f"Error in repo {repo_name}: {str(e)}")

        if not all_results:
            return "No results found."

        return "\n\n".join(all_results)

    def _ensure_repos_downloaded(self, repo_name: str | None = None):
        """
        Helper method to ensure repositories are downloaded to temporary directories.
        Sets self.tmp_dir to a dict mapping repo_names to (tmp_dir, tmp_repo_dir) tuples.

        Args:
            repo_name: If provided, only ensures this specific repo is downloaded.
                      If None, ensures all repos are downloaded.
        """
        if self.tmp_dir is None:
            self.tmp_dir = {}

        downloaded_something = False

        if repo_name:
            # Only download the specified repo if it's not already downloaded
            if repo_name not in self.tmp_dir:
                repo_client = self.context.get_repo_client(
                    repo_name=repo_name, type=self.repo_client_type
                )
                tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()
                self.tmp_dir[repo_name] = (tmp_dir, tmp_repo_dir)
                downloaded_something = True
        else:
            # Download all repos that aren't already downloaded
            repo_names_to_download = [rn for rn in self._get_repo_names() if rn not in self.tmp_dir]

            if repo_names_to_download:

                def download_repo(repo_name):
                    repo_client = self.context.get_repo_client(
                        repo_name=repo_name, type=self.repo_client_type
                    )
                    tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()
                    return repo_name, (tmp_dir, tmp_repo_dir)

                with ThreadPoolExecutor() as executor:
                    future_to_repo = {
                        executor.submit(download_repo, repo_name): repo_name
                        for repo_name in repo_names_to_download
                    }
                    for future in as_completed(future_to_repo):
                        repo_name, repo_dirs = future.result()
                        if repo_name and repo_dirs:
                            self.tmp_dir[repo_name] = repo_dirs

                downloaded_something = True

        # Log whether we downloaded anything new
        append_langfuse_observation_metadata({"repo_download": downloaded_something})

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
            if self.tmp_dir is None or repo_name not in self.tmp_dir:
                continue
            tmp_dir, tmp_repo_dir = self.tmp_dir[repo_name]
            if not tmp_repo_dir:
                continue

            try:
                # Run the find command in the repo directory
                process = subprocess.run(
                    cmd_args,
                    shell=False,
                    cwd=tmp_repo_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if process.returncode != 0:
                    all_results.append(f"Results from {repo_name}: {process.stderr}")
                elif process.stdout:
                    all_results.append(
                        f"Results from {repo_name}:\n------\n{process.stdout}\n------"
                    )
                else:
                    all_results.append(f"Results from {repo_name}: no files found.")
            except Exception as e:
                all_results.append(f"Error in repo {repo_name}: {str(e)}")

        if not all_results:
            return "No results found."

        return "\n\n".join(all_results)

    def get_tools(self, can_access_repos: bool = True):

        tools = [
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
            and (run_request.invoking_user and run_request.invoking_user.id == 3283725)
        ):  # TODO temporary guard for Rohan (@roaga) to test in prod
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
