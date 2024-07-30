import logging
import textwrap

from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.retriever import RetrieverRequest
from seer.automation.autofix.components.retriever_with_reranker import (
    RetrieverWithRerankerComponent,
)
from seer.automation.codebase.code_search import CodeSearcher
from seer.automation.codebase.models import MatchXml
from seer.automation.codebase.utils import cleanup_dir

logger = logging.getLogger(__name__)


class BaseTools:
    context: AutofixContext
    retrieval_top_k: int

    def __init__(self, context: AutofixContext, retrieval_top_k: int = 8):
        self.context = context
        self.retrieval_top_k = retrieval_top_k

    @observe(name="Codebase Search")
    @ai_track(description="Codebase Search")
    def codebase_retriever(self, query: str, intent: str):
        component = RetrieverWithRerankerComponent(self.context)

        output = component.invoke(RetrieverRequest(text=query, intent=intent, top_k=16))

        if not output:
            return "No results found."

        return output.to_xml().to_prompt_str()

    @observe(name="Expand Document")
    @ai_track(description="Expand Document")
    def expand_document(self, input: str, repo_name: str | None = None):
        file_contents = self.context.get_file_contents(input, repo_name=repo_name)

        if repo_name is None:
            client = self.context.get_repo_client(repo_name)
            repo_name = client.repo_name

        self.context.event_manager.add_log(f"Looked at `{input}` in `{repo_name}`")

        if file_contents:
            return file_contents

        return "<document with the provided path not found>"

    @observe(name="List Directory")
    @ai_track(description="List Directory")
    def list_directory(self, path: str, repo_name: str | None = None) -> str:
        """
        Given the path for a directory in this codebase, returns the immediate contents of the directory such as files and direct subdirectories. Does not include nested directories.
        """
        repo_client = self.context.get_repo_client(repo_name=repo_name)

        all_paths = repo_client.get_index_file_set()

        # Normalize the path
        normalized_path = path.strip("/") + "/" if path.strip("/") else ""

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
            self.context.event_manager.add_log(f"Couldn't find anything inside `{path}`")
            return f"<no entries found in directory '{path or '/'}'>"

        # Format the output
        output = []
        if dirs:
            output.append("Directories:")
            output.extend(f"  {d}/" for d in dirs)
        if files:
            if dirs:
                output.append("")  # Add a blank line between dirs and files
            output.append("Files:")
            output.extend(f"  {f}" for f in files)

        self.context.event_manager.add_log(f"Looked through contents of `{path}`")

        joined = "\n".join(output)
        return f"<entries>\n{joined}\n</entries>"

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
        repo_client = self.context.get_repo_client(repo_name=repo_name)

        tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir()

        searcher = CodeSearcher(
            directory=tmp_repo_dir,
            supported_extensions=set(supported_extensions),
            start_path=in_proximity_to,
        )

        results = searcher.search(keyword)

        cleanup_dir(tmp_dir)

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

        self.context.event_manager.add_log(
            f"Searched codebase for `{keyword}`, found {len(file_names)} result(s) in {', '.join(file_names)}"
        )

        return result_str

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
            ),
            FunctionTool(
                name="expand_document",
                fn=self.expand_document,
                description="Given a document path, returns the entire document text.",
                parameters=[
                    {
                        "name": "input",
                        "type": "string",
                        "description": "The document path to expand.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "Optional name of the repository to search in if you know it.",
                    },
                ],
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
            ),
        ]

        if not self.context.skip_loading_codebase:
            tools.append(
                FunctionTool(
                    name="codebase_search",
                    description=textwrap.dedent(
                        """\
                        Search for code snippets in the codebase.
                        - Providing long and detailed queries with entire code snippets will yield better results.
                        - This tool cannot search for code snippets outside the immediate codebase such as in external libraries."""
                    ),
                    parameters=[
                        {
                            "name": "query",
                            "type": "string",
                            "description": "The query to search for.",
                        },
                        {
                            "name": "intent",
                            "type": "string",
                            "description": "The intent of the search, provide a short description of what you're looking for.",
                        },
                    ],
                    fn=self.codebase_retriever,
                )
            )

        return tools
