import fnmatch
import logging
import os
import textwrap

from langfuse.decorators import observe
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
    tmp_dir: str | None = None
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

    @observe(name="Expand Document")
    @ai_track(description="Expand Document")
    def expand_document(self, file_path: str, repo_name: str | None = None):
        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        if repo_name is None:
            client = self.context.get_repo_client(repo_name, self.repo_client_type)
            repo_name = client.repo_name

        self.context.event_manager.add_log(f"Looking at `{file_path}` in `{repo_name}`...")

        if file_contents:
            return file_contents

        # show potential corrected paths if nothing was found here
        other_paths = self._get_potential_abs_paths(file_path, repo_name)
        return f"<document with the provided path not found/>\n{other_paths}".strip()

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
            cleanup_dir(self.tmp_dir)
            self.tmp_dir = None
            self.tmp_repo_dir = None

    @observe(name="Keyword Search")
    @ai_track(description="Keyword Search")
    def keyword_search(
        self,
        keyword: str,
        supported_extensions: list[str],
        repo_name: str | None = None,
        in_proximity_to: str | None = None,
    ):
        """
        Searches for a keyword in the codebase.
        """

        if self.tmp_dir is None or self.tmp_repo_dir is None:
            repo_client = self.context.get_repo_client(
                repo_name=repo_name, type=self.repo_client_type
            )
            tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()

            self.tmp_dir = tmp_dir
            self.tmp_repo_dir = tmp_repo_dir
            append_langfuse_observation_metadata({"keyword_search_download": True})
        else:
            append_langfuse_observation_metadata({"keyword_search_download": False})

        if not self.tmp_repo_dir:
            raise ValueError("tmp_repo_dir is not set")

        searcher = CodeSearcher(
            directory=self.tmp_repo_dir,
            supported_extensions=set(supported_extensions),
            start_path=in_proximity_to,
        )

        results = searcher.search(keyword)

        self.context.event_manager.add_log(
            f"Searched codebase for `{keyword}`, found {len(results)} result(s)."
        )

        if not results:
            return "No results found."

        result_str = ""
        file_names = []
        for result in results:
            for match in result.matches:
                match_xml = MatchXml(
                    path=result.relative_path,
                    context=match.context,
                )
                file_names.append(f"`{result.relative_path}`")
                result_str += f"{match_xml.to_prompt_str()}\n\n"

        return result_str

    @observe(name="File Search")
    @ai_track(description="File Search")
    def file_search(
        self,
        filename: str,
        repo_name: str | None = None,
    ):
        """
        Given a filename with extension returns the list of locations where a file with the name is found.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name, type=self.repo_client_type)
        all_paths = repo_client.get_index_file_set()
        found = [path for path in all_paths if os.path.basename(path) == filename]

        self.context.event_manager.add_log(f"Searching for file `{filename}` in `{repo_name}`...")

        if len(found) == 0:
            return f"no file with name {filename} found in repository"

        found = sorted(found)

        return ",".join(found)

    @observe(name="File Search Wildcard")
    @ai_track(description="File Search Wildcard")
    def file_search_wildcard(
        self,
        pattern: str,
        repo_name: str | None = None,
    ):
        """
        Given a filename pattern with wildcards, returns the list of file paths that match the pattern.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name, type=self.repo_client_type)
        all_paths = repo_client.get_index_file_set()
        found = [path for path in all_paths if fnmatch.fnmatch(path, pattern)]

        self.context.event_manager.add_log(
            f"Searching for files with pattern `{pattern}` in `{repo_name}`..."
        )

        if len(found) == 0:
            return f"No files matching pattern '{pattern}' found in repository"

        found = sorted(found)

        return "\n".join(found)

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
    def search_google(self, question: str, llm_client: LlmClient = injected):
        """
        Searches Google to answer a question.
        """
        self.context.event_manager.add_log(f'Googling "{question}"...')
        return llm_client.generate_text_from_web_search(
            prompt=question, model=GeminiProvider(model_name="gemini-2.0-flash-exp")
        )

    def get_tools(self):
        tools = [
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
                        "description": "Optional name of the repository to search in if you know it.",
                    },
                ],
                required=["file_path"],
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
                        "name": "repo_name",
                        "type": "string",
                        "description": "Optional name of the repository to search in if you know it.",
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
                description="Searches for a file in the codebase.",
                parameters=[
                    {
                        "name": "filename",
                        "type": "string",
                        "description": "The file to search for.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "Optional name of the repository to search in if you know it.",
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
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "Optional name of the repository to search in if you know it.",
                    },
                ],
                required=["pattern"],
            ),
            FunctionTool(
                name="search_google",
                fn=self.search_google,
                description="Searches the web with Google and returns the answer to a question.",
                parameters=[
                    {
                        "name": "question",
                        "type": "string",
                        "description": "The question you want to answer.",
                    },
                ],
                required=["question"],
            ),
        ]

        if (
            isinstance(self.context, AutofixContext)
            and not self.context.state.get().request.options.disable_interactivity
        ):
            tools.append(
                FunctionTool(
                    name="ask_a_question",
                    fn=self.ask_user_question,
                    description="Asks your team members a quick question.",
                    parameters=[
                        {
                            "name": "question",
                            "type": "string",
                            "description": "The question you want to ask your team.",
                        }
                    ],
                    required=["question"],
                )
            )

        return tools
