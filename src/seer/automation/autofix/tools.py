import fnmatch
import json
import logging
import os
import textwrap
import time

from codegen import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.detached_symbols.function_call import FunctionCall
from codegen.sdk.core.function import Function
from codegen.sdk.core.statements.statement import Statement
from codegen.sdk.enums import SymbolType
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

    codebases: dict[str, Codebase] = {}

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
        self._setup_tmp_dir_and_download()
        if self.tmp_dir:
            for repo_name, (tmp_dir, tmp_repo_dir) in self.tmp_dir.items():
                start_time = time.time()
                self.codebases[repo_name] = Codebase(tmp_repo_dir)
                logger.info(f"Codebase initialization took {time.time() - start_time:.2f} seconds")
        else:
            raise ValueError("No tmp_dir found")

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

    def _setup_tmp_dir_and_download(self):
        repo_names = self._get_repo_names()
        if self.tmp_dir is None:
            tmp_dirs = {}
            for repo_name in repo_names:
                repo_client = self.context.get_repo_client(
                    repo_name=repo_name, type=self.repo_client_type
                )
                tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()
                tmp_dirs[repo_name] = (tmp_dir, tmp_repo_dir)

                self._initialize_git_repo(tmp_repo_dir)

            self.tmp_dir = tmp_dirs  # Store all tmp dirs

    def _initialize_git_repo(self, repo_dir: str):
        """Initialize a Git repository in the specified directory."""
        import subprocess

        try:
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=repo_dir, check=True)

            # Add all files to git
            subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)

            # Create initial commit
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=repo_dir,
                check=True,
                env={
                    **os.environ,
                    "GIT_COMMITTER_NAME": "Autofix",
                    "GIT_COMMITTER_EMAIL": "autofix@example.com",
                    "GIT_AUTHOR_NAME": "Autofix",
                    "GIT_AUTHOR_EMAIL": "autofix@example.com",
                },
            )

            logger.info(f"Git repository initialized in {repo_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Git repository: {e}")
        except Exception as e:
            logger.error(f"Error initializing Git repository: {e}")

    def _get_symbol(self, query: str, repo_name: str):
        segments = query.split(".")

        if len(segments) > 2:
            return None, f"Symbol `{query}` not found"

        symbol = self.codebases[repo_name].get_symbol(symbol_name=segments[0], optional=True)

        if not symbol:
            return None, f"Symbol `{segments[0]}` not found"

        if isinstance(symbol, Class) and len(segments) == 2:
            method = symbol.get_method(segments[1])
            if not method:
                attribute = symbol.get_attribute(segments[1])
                if not attribute:
                    return None, f"Symbol `{segments[1]}` not found in class `{symbol.name}`"
                symbol = attribute
            else:
                symbol = method

        return symbol, None

    @observe(name="Find Definition")
    @ai_track(description="Semantic File Search")
    def find_definition(self, query: str):
        repo_names = self._get_repo_names()
        for repo_name in repo_names:
            symbol, error = self._get_symbol(query, repo_name)

            if error:
                return json.dumps({"error": error or "Symbol not found"}, indent=2)

            if symbol:
                return json.dumps(
                    {
                        "file_path": symbol.filepath,
                        "source": symbol.source,
                    },
                    indent=2,
                )

        return json.dumps({"error": "Symbol not found"}, indent=2)

    @observe(name="Find Call Sites")
    @ai_track(description="Find Call Sites")
    def find_call_sites(self, query: str):
        def get_parent_function(call_site: FunctionCall):
            parent_function = call_site.parent_function

            remaining_iterations = 5

            can_be_parent = False
            try:
                parent_function.extended_source
                can_be_parent = True
            except:
                pass

            while not can_be_parent and remaining_iterations > 0:
                try:
                    parent_function = parent_function.parent_function
                    parent_function.extended_source
                    can_be_parent = True
                except:
                    pass

                remaining_iterations -= 1

            return parent_function

        repo_names = self._get_repo_names()
        for repo_name in repo_names:
            symbol, error = self._get_symbol(query, repo_name)

            if error:
                return json.dumps({"error": error or "Symbol not found"}, indent=2)

            if symbol:
                if not isinstance(symbol, Function) and not isinstance(symbol, Class):
                    return json.dumps({"error": "Symbol is not a function or class"}, indent=2)

                call_sites = []
                for call_site in symbol.call_sites:
                    invoking_parent_source = None
                    try:
                        invoking_parent = get_parent_function(call_site)
                        invoking_parent_source = invoking_parent.extended_source
                    except Exception as e:
                        print(e)

                    call_sites.append(
                        {
                            "file_path": call_site.filepath,
                            "call_location": call_site.source,
                            "source": invoking_parent_source,
                        }
                    )

                return json.dumps({"call_sites": call_sites}, indent=2)

        return json.dumps({"error": "Symbol not found"}, indent=2)

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
            files_per_repo[repo_name] = "\n".join(sorted(valid_file_paths))

        self.context.event_manager.add_log(f'Searching for "{query}"...')

        class FilePath(BaseModel):
            file_path: str
            repo_name: str

        all_valid_paths = "\n".join(
            [
                f"FILES IN REPO {repo_name}:\n{files_per_repo[repo_name]}\n------------"
                for repo_name in repo_names
            ]
        )
        prompt = textwrap.dedent(
            """
            I'm searching for the file in this codebase that contains {query}. Please pick the most relevant file from the following list:
            ------------
            {valid_file_paths}
            """
        ).format(query=query, valid_file_paths=all_valid_paths)

        response = llm_client.generate_structured(
            prompt=prompt,
            model=GeminiProvider(model_name="gemini-2.0-flash-001"),
            response_format=FilePath,
        )
        result = response.parsed
        file_path = result.file_path if result else None
        repo_name = result.repo_name if result else None
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

        self.codebases = {}

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

        all_results = []

        repo_names = self._get_repo_names()

        if not self.tmp_dir:
            return "No codebase found. Please try again later."

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
    def search_google(self, question: str, llm_client: LlmClient = injected):
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
                    FunctionTool(
                        name="find_definition",
                        fn=self.find_definition,
                        description="Finds the definition of a symbol such as a function, class, or variable in the codebase.",
                        parameters=[
                            {
                                "name": "query",
                                "type": "string",
                                "description": "The symbol to find the definition of, for class methods, include the class name in the query. For example, 'MyClass.my_method'.",
                            },
                        ],
                        required=["query"],
                    ),
                    FunctionTool(
                        name="find_call_sites",
                        fn=self.find_call_sites,
                        description="Finds all the places that a symbol such as a function or class is called in the codebase.",
                        parameters=[
                            {
                                "name": "query",
                                "type": "string",
                                "description": "The symbol to find the call sites of, for class methods, include the class name in the query. For example, 'MyClass.my_method'.",
                            },
                        ],
                        required=["query"],
                    ),
                ]
            )

        # TODO: Re-enable soon once we have it ask less bad questions
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
