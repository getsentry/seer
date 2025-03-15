import fnmatch
import logging
import os
import textwrap
from typing import Literal, TypeAlias

from langfuse.decorators import observe
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.codebase.code_search import CodeSearcher
from seer.automation.codebase.models import MatchXml
from seer.automation.codebase.repo_client import RepoClientType
from seer.automation.codebase.utils import cleanup_dir
from seer.automation.codegen.codegen_context import CodegenContext
from seer.dependency_injection import inject, injected
from seer.langfuse import append_langfuse_observation_metadata

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

        if len(repo_names) < 100:  # structured output can't handle too many in Literal
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

    @observe(name="List Directory")
    @ai_track(description="List Directory")
    def list_directory(self, path: str, repo_name: str | None = None) -> str:
        """
        Given the path for a directory in this codebase, returns the immediate contents of the directory such as files and direct subdirectories. Does not include nested directories.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name, type=self.repo_client_type)
        all_paths = repo_client.get_index_file_set()
        normalized_path = self._normalize_path(path)

        # Filter paths to include only those directly under the specified path + remove duplicates and sort
        unique_direct_children = sorted(
            set(
                p[len(normalized_path) :].split("/")[0]
                for p in all_paths
                if p.startswith(normalized_path) and p != normalized_path
            )
        )

        # Separate directories and files
        dirs, files = self._separate_dirs_and_files(
            normalized_path, unique_direct_children, all_paths
        )

        if not dirs and not files:
            # show potential corrected paths if nothing was found here
            other_paths = self._get_potential_abs_paths(path, repo_name)
            return f"<no entries found in directory '{path or '/'}'/>\n{other_paths}".strip()

        self.context.event_manager.add_log(f"Looking at contents of `{path}` in `{repo_name}`...")

        joined = self._format_list_directory_output(dirs, files)
        return f"<entries>\n{joined}\n</entries>"

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

    def _format_list_directory_output(self, dirs: list[str], files: list[str]) -> str:
        output = []
        if dirs:
            output.append("Directories:")
            output.extend(f"  {d}/" for d in dirs)
        if files:
            if dirs:
                output.append("")  # Add a blank line between dirs and files
            output.append("Files:")
            output.extend(f"  {f}" for f in files)

        joined = "\n".join(output)
        return joined

    def _separate_dirs_and_files(
        self, parent_path: str, direct_children: list[str], all_paths: set
    ) -> tuple[list[str], list[str]]:
        dirs = []
        files = []
        for child in direct_children:
            full_path = f"{parent_path}{child}"
            if any(p.startswith(full_path + "/") for p in all_paths):
                dirs.append(child)
            else:
                files.append(child)
        return dirs, files

    def cleanup(self):
        if self.tmp_dir:
            # Clean up all tmp dirs
            for tmp_dir, _ in self.tmp_dir.values():
                cleanup_dir(tmp_dir)
            self.tmp_dir = None

    @observe(name="Keyword Search")
    @ai_track(description="Keyword Search")
    def keyword_search(
        self,
        keyword: str,
        supported_extensions: list[str],
        in_proximity_to: str | None = None,
    ):
        """
        Searches for a keyword in the codebase.
        """
        repo_names = self._get_repo_names()
        all_results = []

        if self.tmp_dir is None:
            tmp_dirs = {}
            for repo_name in repo_names:
                repo_client = self.context.get_repo_client(
                    repo_name=repo_name, type=self.repo_client_type
                )
                tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()
                tmp_dirs[repo_name] = (tmp_dir, tmp_repo_dir)

            self.tmp_dir = tmp_dirs  # Store all tmp dirs
            append_langfuse_observation_metadata({"keyword_search_download": True})
        else:
            append_langfuse_observation_metadata({"keyword_search_download": False})

        for repo_name in repo_names:
            tmp_dir, tmp_repo_dir = self.tmp_dir[repo_name]
            if not tmp_repo_dir:
                continue

            searcher = CodeSearcher(
                directory=tmp_repo_dir,
                supported_extensions=set(supported_extensions),
                start_path=in_proximity_to,
            )

            results = searcher.search(keyword)
            if results:
                for result in results:
                    for match in result.matches:
                        match_xml = MatchXml(
                            path=result.relative_path,
                            repo_name=repo_name,
                            context=match.context,
                        )
                        all_results.append(match_xml.to_prompt_str())

        self.context.event_manager.add_log(
            f"Searched codebase for `{keyword}`, found {len(all_results)} result(s)."
        )

        if not all_results:
            return "No results found."

        return "\n\n".join(all_results)

    @observe(name="File Search")
    @ai_track(description="File Search")
    def file_search(
        self,
        filename: str,
    ):
        """
        Given a filename with extension returns the list of locations where a file with the name is found.
        """
        repo_names = self._get_repo_names()
        found_files = ""

        for repo_name in repo_names:
            repo_client = self.context.get_repo_client(
                repo_name=repo_name, type=self.repo_client_type
            )
            all_paths = repo_client.get_index_file_set()
            found = [
                path for path in all_paths if os.path.basename(path).lower() == filename.lower()
            ]
            if found:
                found_files += f"\n FILES IN REPO {repo_name}:\n"
                found_files += "\n".join([f"  {path}" for path in sorted(found)])

        self.context.event_manager.add_log(f"Searching for file `{filename}`...")

        if len(found_files) == 0:
            return f"no file with name {filename} found in any repository"

        return found_files

    @observe(name="File Search Wildcard")
    @ai_track(description="File Search Wildcard")
    def file_search_wildcard(
        self,
        pattern: str,
    ):
        """
        Given a filename pattern with wildcards, returns the list of file paths that match the pattern.
        """
        repo_names = self._get_repo_names()
        found_files = ""

        for repo_name in repo_names:
            repo_client = self.context.get_repo_client(
                repo_name=repo_name, type=self.repo_client_type
            )
            all_paths = repo_client.get_index_file_set()
            found = [path for path in all_paths if fnmatch.fnmatch(path, pattern)]
            if found:
                found_files += f"\n FILES IN REPO {repo_name}:\n"
                found_files += "\n".join([f"  {path}" for path in sorted(found)])

        self.context.event_manager.add_log(f"Searching for files with pattern `{pattern}`...")

        if len(found_files) == 0:
            return f"No files matching pattern '{pattern}' found in any repository"

        return found_files

    @observe(name="Ask User Question")
    @ai_track(description="Ask User Question")
    def ask_user_question(self, question: str):
        """
        Sends a question to the user on the frontend and waits for a response before continuing.
        """
        if isinstance(self.context, AutofixContext):
            self.context.event_manager.ask_user_question(question)

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
                        name="list_directory",
                        fn=self.list_directory,
                        description="Given the path for a directory in this codebase, returns the immediate contents of the directory such as files and direct subdirectories. Does not include nested directories.",
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
                        name="keyword_search",
                        fn=self.keyword_search,
                        description="Searches for a keyword in the codebase.",
                        parameters=[
                            {
                                "name": "keyword",
                                "type": "string",
                                "description": "The keyword to search for.",
                            },
                            {
                                "name": "supported_extensions",
                                "type": "array",
                                "description": "The str[] of supported extensions to search in. Include the dot in the extension. For example, ['.py', '.js'].",
                                "items": {"type": "string"},
                            },
                            {
                                "name": "in_proximity_to",
                                "type": "string",
                                "description": "Optional path to search in proximity to, the results will be ranked based on proximity to this path.",
                            },
                        ],
                        required=["keyword", "supported_extensions"],
                    ),
                    FunctionTool(
                        name="file_search",
                        fn=self.file_search,
                        description="Given a filename with extension returns the list of locations where a file with the name is found.",
                        parameters=[
                            {
                                "name": "filename",
                                "type": "string",
                                "description": "The filename with extension to search for.",
                            },
                        ],
                        required=["filename"],
                    ),
                    FunctionTool(
                        name="file_search_wildcard",
                        fn=self.file_search_wildcard,
                        description="Searches for files in a folder using a wildcard pattern.",
                        parameters=[
                            {
                                "name": "pattern",
                                "type": "string",
                                "description": "The wildcard pattern to match files.",
                            },
                        ],
                        required=["pattern"],
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

        return tools
