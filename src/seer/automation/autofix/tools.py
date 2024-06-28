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
from seer.automation.autofix.components.snippet_replacement import (
    SnippetReplacementComponent,
    SnippetReplacementRequest,
)
from seer.automation.autofix.utils import find_original_snippet
from seer.automation.models import FileChange

logger = logging.getLogger("autofix")


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

        self.context.event_manager.add_log(
            f"Taking a look at the document at {input} in {repo_name}."
        )

        if file_contents:
            return file_contents

        return "<document with the provided path not found>"

    @observe(name="List Directory")
    @ai_track(description="List Directory")
    def list_directory(self, path: str, repo_name: str | None = None):
        repo_client = self.context.get_repo_client(repo_name=repo_name)

        all_paths = repo_client.get_index_file_set(repo_client.get_default_branch_head_sha())

        paths = [p for p in all_paths if p.startswith(path)]

        if not paths:
            return f"<no paths found in directory {path}>"

        joined = "\n".join(paths)
        paths_str = f"<paths>\n{joined}\n</paths>"

        return paths_str

    def get_tools(self):
        tools = [
            FunctionTool(
                name="list_directory",
                fn=self.list_directory,
                description="Given the path for a directory in this codebase, returns the paths of the directory.",
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
        ]

        # if not self.context.skip_loading_codebase:
        #     tools.append(
        #         FunctionTool(
        #             name="codebase_search",
        #             description=textwrap.dedent(
        #                 """\
        #             Search for code snippets in the codebase.
        #             - Providing long and detailed queries with entire code snippets will yield better results.
        #             - This tool cannot search for code snippets outside the immediate codebase such as in external libraries."""
        #             ),
        #             parameters=[
        #                 {
        #                     "name": "query",
        #                     "type": "string",
        #                     "description": "The query to search for.",
        #                 },
        #                 {
        #                     "name": "intent",
        #                     "type": "string",
        #                     "description": "The intent of the search, provide a short description of what you're looking for.",
        #                 },
        #             ],
        #             fn=self.codebase_retriever,
        #         )
        #     )

        return tools


class CodeActionTools(BaseTools):
    snippet_matching_threshold = 0.8
    chunk_padding = 16

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @observe(name="Store File Change")
    @ai_track(description="Store File Change")
    def store_file_change(self, repo_name: str, file_change: FileChange):
        """
        Stores a file change to a codebase index.
        This function exists mainly to be traceable in Langsmith.
        """
        with self.context.state.update() as cur:
            repo_client = self.context.get_repo_client(repo_name)
            cur.codebases[repo_client.repo_external_id].file_changes.append(file_change)

        if not self.context.skip_loading_codebase:
            repo_id = self.context.get_repo_id_from_name(repo_name)
            if repo_id:
                codebase = self.context.get_codebase(repo_id)
                if codebase:
                    codebase.store_file_change(file_change)
                else:
                    # Exception for sentry to log but we don't inform the LLM
                    logger.exception(
                        ValueError(
                            f"Codebase for repo id {repo_id} of repo name {repo_name} not found."
                        ),
                        exc_info=True,
                    )
            else:
                # Exception for sentry to log but we don't inform the LLM
                logger.exception(ValueError(f"Repo {repo_name} not found."), exc_info=True)

        self.context.event_manager.add_log(
            f"Made a code change in {file_change.path} in {repo_name}."
        )

    @observe(name="Replace Snippet")
    @ai_track(description="Replace Snippet")
    def replace_snippet_with(
        self,
        file_path: str,
        repo_name: str,
        reference_snippet: str,
        replacement_snippet: str,
        commit_message: str,
    ):
        """
        Replaces a snippet with the provided replacement.
        """
        logger.debug(
            f"[CodeActionTools.replace_snippet_with] Replacing snippet\n```\n{reference_snippet}\n```\n with \n```\n{replacement_snippet}\n```\nin {file_path} in repo {repo_name}"
        )

        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        if not file_contents:
            raise FileNotFoundError("File not found or it was deleted in a previous action.")

        result = find_original_snippet(
            reference_snippet, file_contents, threshold=self.snippet_matching_threshold
        )

        if not result:
            raise Exception("Reference snippet not found. Try again with an exact match.")

        original_snippet, snippet_start_line, snippet_end_line = result

        lines = file_contents.splitlines()
        chunk_lines = lines[
            max(0, snippet_start_line - self.chunk_padding) : min(
                len(lines), snippet_end_line + self.chunk_padding
            )
        ]
        chunk = "\n".join(chunk_lines).strip("\n")

        if not original_snippet:
            raise Exception("Reference snippet not found. Try again with an exact match.")

        output = SnippetReplacementComponent(self.context).invoke(
            SnippetReplacementRequest(
                reference_snippet=original_snippet,
                replacement_snippet=replacement_snippet,
                chunk=chunk,
                commit_message=commit_message,
            )
        )

        if not output:
            raise Exception("Snippet replacement failed.")

        # Add a trailing newline in the reference snippet, this is because we stripped all newlines from the chunk originally, we should add the trailing one back in.
        reference_snippet = chunk + "\n"
        new_snippet = output.snippet
        # Add a trailing snippet to the new snippet to match the reference snippet if there isn't already one.
        if not new_snippet.endswith("\n"):
            new_snippet += "\n"

        self.store_file_change(
            repo_name,
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=reference_snippet,
                new_snippet=new_snippet,
                description=commit_message,
            ),
        )

        return f"success: Resulting code after replacement:\n```\n{output.snippet}\n```\n"

    @observe(name="Delete Snippet")
    @ai_track(description="Delete Snippet")
    def delete_snippet(self, file_path: str, repo_name: str, snippet: str, commit_message: str):
        """
        Deletes a snippet.
        """
        logger.debug(
            f"[CodeActionTools.delete_snippet] Deleting snippet {snippet} in {file_path} in repo {repo_name}"
        )

        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        if not file_contents:
            raise FileNotFoundError("File not found or it was deleted in a previous action.")

        original_snippet: str | None = None
        if snippet in file_contents:
            original_snippet = snippet
        else:
            result = find_original_snippet(
                snippet, file_contents, threshold=self.snippet_matching_threshold
            )

            if result:
                original_snippet, snippet_start_line, snippet_end_line = result

        logger.debug("Exact snippet:")
        logger.debug(f'"{snippet}"')

        if not original_snippet:
            raise Exception("Reference snippet not found. Try again with an exact match.")
        file_change = FileChange(
            change_type="delete",
            path=file_path,
            description=commit_message,
            reference_snippet=original_snippet,
            new_snippet="",
        )
        self.store_file_change(
            repo_name,
            file_change,
        )

        return f"success; New file contents for `{file_path}`: \n\n```\n{file_change.apply(file_contents)}\n```"

    @observe(name="Create File")
    @ai_track(description="Create File")
    def create_file(self, file_path: str, repo_name: str, snippet: str, commit_message: str):
        """
        Creates a file with the provided snippet.
        """
        logger.debug(
            f"[CodeActionTools.create_file] Creating file {file_path} with snippet {snippet} in {repo_name}"
        )

        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        if file_contents:
            raise FileExistsError(f"File `{file_path}` already exists.")

        self.store_file_change(
            repo_name,
            FileChange(
                change_type="create",
                path=file_path,
                new_snippet=snippet,
                description=commit_message,
            ),
        )

        return "success"

    @observe(name="Delete File")
    @ai_track(description="Delete File")
    def delete_file(self, file_path: str, repo_name: str, commit_message: str):
        """
        Deletes a file.
        """
        logger.debug(f"[CodeActionTools.delete_file] Deleting file {file_path} in {repo_name}")

        file_contents = self.context.get_file_contents(file_path, repo_name=repo_name)

        if not file_contents:
            raise FileNotFoundError(f"File `{file_path}` not found.")

        self.store_file_change(
            repo_name,
            FileChange(change_type="delete", path=file_path, description=commit_message),
        )

        return "success"

    def get_retrospection_tools(self):
        return super().get_tools()

    def get_tools(self):
        return super().get_tools() + [
            FunctionTool(
                name="replace_snippet_with",
                description=textwrap.dedent(
                    """\
                    Use this as the primary tool to write code changes to a file.

                    Replaces a snippet in a file with the provided replacement.
                    - The snippet must be an exact match.
                    - The replacement can be any string.
                    - The reference snippet must be an entire line, not just a substring of a line. It should also include the indentation and spacing.
                    - Indentation and spacing must be included in the replacement snippet."""
                ),
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to modify.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "reference_snippet",
                        "type": "string",
                        "description": "The snippet to replace.",
                    },
                    {
                        "name": "replacement_snippet",
                        "type": "string",
                        "description": "The replacement for the snippet.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.replace_snippet_with,
            ),
            FunctionTool(
                name="delete_snippet",
                description=textwrap.dedent(
                    """\
                    Deletes a snippet in a file.
                    - The snippet must be an exact match."""
                ),
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to modify.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "snippet",
                        "type": "string",
                        "description": "The snippet to delete.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.delete_snippet,
            ),
            FunctionTool(
                name="create_file",
                description="""Creates a file with the provided snippet.""",
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to create.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "snippet",
                        "type": "string",
                        "description": "The snippet to insert.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.create_file,
            ),
            FunctionTool(
                name="delete_file",
                description="Deletes a file.",
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to delete.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.delete_file,
            ),
        ]
