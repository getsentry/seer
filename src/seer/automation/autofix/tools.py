import logging
from typing import Dict, Set

from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import MetadataMode

from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.agent_context import AgentContext
from seer.automation.autofix.types import FileChange
from seer.automation.autofix.utils import find_original_snippet

logger = logging.getLogger("autofix")


class BaseTools:
    context: AgentContext
    retriever: VectorIndexRetriever

    retrieved_paths: Set[str]
    expanded_paths: Set[str]

    def __init__(self, context: AgentContext):
        self.context = context
        self.retriever = VectorIndexRetriever(index=self.context.index, similarity_top_k=4)
        self.retrieved_paths = set()
        self.expanded_paths = set()

    def codebase_retriever(self, query: str):
        nodes = self.retriever.retrieve(query)

        content = ""
        for node in nodes:
            self.retrieved_paths.add(node.metadata["file_path"])

            node_copy = node.copy()
            # node_copy.text_template = "{metadata_str}\n{content}"
            # node_copy.metadata_template = "{key} = {value}"
            content += node_copy.get_content(MetadataMode.LLM) + "\n\n"
        return content

    def _get_document(self, file_path: str):
        for document in self.context.documents:
            if file_path in document.metadata["file_path"]:
                return document

        return None

    def expand_document(self, input: str):
        document = self._get_document(input)

        if document:
            self.expanded_paths.add(document.metadata["file_path"])
            return document.text

        return "<document with the provided path not found>"

    def get_tools(self):
        return [
            FunctionTool(
                name="codebase_search",
                description="""Search for code snippets.
- You can search for code using either a code snippet or the path.
- The codebase is large, so you will need to be very specific with your query.
- If the path contains relative paths such as ../, you will need to remove them.
- If "code" in "file" search does not work, try searching just the code snippet.

Example Queries:
- Search for a code snippet: "foo"
- Search for a file: "sentry/data/issueTypeConfig/index.tsx"
- Search for a function: "getIssueTypeConfig("
""",
                parameters=[
                    {
                        "name": "query",
                        "type": "string",
                        "description": "The query to search for.",
                    }
                ],
                fn=self.codebase_retriever,
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
                    }
                ],
            ),
        ]


class CodeActionTools(BaseTools):
    context: AgentContext
    base_sha: str
    file_changes: Dict[str, FileChange]

    _snippet_matching_threshold = 0.9

    def __init__(
        self,
        context: AgentContext,
        base_sha: str,
        verbose: bool = False,
    ):
        super().__init__(context)

        self.context = context
        self.base_sha = base_sha
        self.verbose = verbose
        self.file_changes = {}

    def _get_latest_file_contents(self, file_path: str):
        if file_path not in self.file_changes:
            logger.debug(
                f"Getting file contents from Github for file_path: {file_path} from sha {self.base_sha}"
            )
            contents = self.context.get_file_contents(file_path, self.base_sha)

            return contents

        logger.debug(f"Getting file contents from memory for file_path: {file_path}")

        return self.file_changes[file_path].contents

    def expand_document(self, input: str):
        if input in self.file_changes:
            return self.file_changes[input].contents

        return super().expand_document(input)

    def replace_snippet_with(
        self, file_path: str, reference_snippet: str, replacement_snippet: str, commit_message: str
    ):
        """
        Replaces a snippet with the provided replacement.
        """
        logger.debug(
            f"[CodeActionTools.replace_snippet_with] Replacing snippet {reference_snippet} with {replacement_snippet} in {file_path}"
        )

        file_contents = self._get_latest_file_contents(file_path)

        if not file_contents:
            raise Exception("File not found.")

        logger.debug("Exact snippet:")
        logger.debug(f'"{reference_snippet}"')

        original_snippet: str | None = None
        if reference_snippet in file_contents:
            original_snippet = reference_snippet
        else:
            original_snippet = find_original_snippet(
                reference_snippet, file_contents, threshold=self._snippet_matching_threshold
            )

        if not original_snippet:
            raise Exception("Reference snippet not found. Try again with an exact match.")

        new_contents = file_contents.replace(original_snippet, replacement_snippet)

        self.context._update_document(file_path, new_contents)

        original_contents = file_contents
        if file_path in self.file_changes:
            original_contents = self.file_changes[file_path].original_contents

        self.file_changes[file_path] = FileChange(
            change_type="edit",
            path=file_path,
            contents=new_contents,
            description=commit_message,
            original_contents=original_contents,
        )

        return f"success; New file contents for `{file_path}`: \n\n```\n{new_contents}\n```"

    def delete_snippet(self, file_path: str, snippet: str, commit_message: str):
        """
        Deletes a snippet.
        """
        logger.debug(f"[CodeActionTools.delete_snippet] Deleting snippet {snippet} in {file_path}")

        file_contents = self._get_latest_file_contents(file_path)

        if not file_contents:
            raise Exception("File not found.")

        original_snippet: str | None = None
        if snippet in file_contents:
            original_snippet = snippet
        else:
            original_snippet = find_original_snippet(
                snippet, file_contents, threshold=self._snippet_matching_threshold
            )

        logger.debug("Exact snippet:")
        logger.debug(f'"{snippet}"')

        if not original_snippet:
            raise Exception("Reference snippet not found. Try again with an exact match.")

        new_contents = file_contents.replace(original_snippet, "")

        self.context._update_document(file_path, new_contents)

        original_contents = file_contents
        if file_path in self.file_changes:
            original_contents = self.file_changes[file_path].original_contents

        self.file_changes[file_path] = FileChange(
            change_type="edit",
            path=file_path,
            contents=new_contents,
            description=commit_message,
            original_contents=original_contents,
        )

        return f"success; New file contents for `{file_path}`: \n\n```\n{new_contents}\n```"

    def insert_snippet(
        self, file_path: str, reference_snippet: str, snippet: str, commit_message: str
    ):
        """
        Inserts a snippet after the reference snippet.
        """

        logger.debug(
            f"[CodeActionTools.insert_snippet] Inserting snippet {snippet} after {reference_snippet} in {file_path}"
        )

        file_contents = self._get_latest_file_contents(file_path)

        if not file_contents:
            raise Exception("File not found.")

        original_snippet = find_original_snippet(
            reference_snippet, file_contents, threshold=self._snippet_matching_threshold
        )

        logger.debug("Exact reference snippet:")
        logger.debug(f'"{reference_snippet}"')

        if not original_snippet:
            raise Exception("Reference snippet not found. Try again with an exact match.")

        new_contents = file_contents.replace(original_snippet, original_snippet + "\n" + snippet)

        self.context._update_document(file_path, new_contents)

        original_contents = file_contents
        if file_path in self.file_changes:
            original_contents = self.file_changes[file_path].original_contents

        self.file_changes[file_path] = FileChange(
            change_type="edit",
            path=file_path,
            contents=new_contents,
            description=commit_message,
            original_contents=original_contents,
        )

        return f"success; New file contents for `{file_path}`: \n\n```\n{new_contents}\n```"

    def create_file(self, file_path: str, snippet: str, commit_message: str):
        """
        Creates a file with the provided snippet.
        """
        logger.debug(
            f"[CodeActionTools.create_file] Creating file {file_path} with snippet {snippet}"
        )

        self.context._update_document(file_path, snippet)

        self.file_changes[file_path] = FileChange(
            change_type="edit",
            path=file_path,
            contents=snippet,
            description=commit_message,
            original_contents=None,
        )

        return "success"

    def delete_file(self, file_path: str, commit_message: str):
        """
        Deletes a file.
        """
        logger.debug(f"[CodeActionTools.delete_file] Deleting file {file_path}")

        self.file_changes[file_path] = FileChange(
            change_type="delete", path=file_path, description=commit_message
        )

        self.context._update_document(file_path, None)

        return "success"

    def get_retrospection_tools(self):
        return super().get_tools()

    def get_tools(self):
        return super().get_tools() + [
            FunctionTool(
                name="replace_snippet_with",
                description="""Replaces a snippet in a file with the provided replacement.
- The snippet must be an exact match.
- The replacement can be any string.
- The reference snippet must be an entire line, not just a substring of a line.
- Indentation and spacing must be included in the replacement snippet.""",
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to modify.",
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
                description="""Deletes a snippet in a file.
- The snippet must be an exact match.""",
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to modify.",
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
                name="insert_snippet",
                description="""Inserts a snippet on a new line directly after reference snippet in a file.
- The snippet must be an exact match.
- The reference snippet must be an exact match.
- Indentation and spacing must be included in the snippet to insert.""",
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to modify.",
                    },
                    {
                        "name": "reference_snippet",
                        "type": "string",
                        "description": "The reference snippet to insert after.",
                    },
                    {
                        "name": "snippet",
                        "type": "string",
                        "description": "The snippet to insert. This snippet will be on a new line after the reference snippet.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.insert_snippet,
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
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.delete_file,
            ),
        ]
