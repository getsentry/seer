import hashlib
import logging
import textwrap
import time

import tree_sitter_languages
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tree_sitter import Node

from seer.automation.codebase.models import Document, DocumentChunk
from seer.utils import class_method_lru_cache

logger = logging.getLogger("autofix")


def get_indent_start_byte(node: Node, root: Node) -> int:
    if node.start_byte == 0:
        return 0
    newline_index = root.text.rfind(b"\n", 0, node.start_byte)
    if newline_index == -1:
        # No newline character found, return start_byte
        return node.start_byte
    line_start_byte = newline_index + 1
    # Check if there are only spaces or tabs before the newline
    if all(c in b" \t" for c in root.text[line_start_byte : node.start_byte]):
        return line_start_byte
    else:
        return node.start_byte


def index_with_node_type(node_type: str, node: Node, recursive=True):
    for i, child in enumerate(node.children):
        if child.type == node_type:
            return i, child
        if recursive:
            descendant_result = index_with_node_type(node_type, child, recursive=True)
            if descendant_result:
                return i, descendant_result[1]
    return None


def first_child_with_type(node_types: set[str], node: Node):
    for child in node.children:
        if child.type in node_types:
            return child
    return None


class ParentDeclaration(BaseModel):
    id: int
    declaration_byte_start: int
    declaration_byte_end: int
    content_indent: str

    declaration_nodes: list[Node]

    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other):
        if isinstance(other, ParentDeclaration):
            return self.id == other.id
        return False

    def __hash__(self):
        return self.id


class TempChunk(BaseModel):
    nodes: list[Node]
    parent_declarations: list[ParentDeclaration]

    class Config:
        arbitrary_types_allowed = True

    def get_content(self, root: Node) -> str:
        group_text = ""
        # I think it doesn't matter if we miss the spacing between the last chunk and this one
        last_end = get_indent_start_byte(self.nodes[0], root)
        for node in self.nodes:
            group_text += root.text[last_end : node.end_byte].decode("utf-8")
            last_end = node.end_byte
        return group_text

    def get_context(self, root: Node) -> str:
        context_text = ""
        for declaration in self.parent_declarations:
            context_text += textwrap.dedent(
                """\
                {declaration}
                {content_indent}...
                """
            ).format(
                declaration=root.text[
                    declaration.declaration_byte_start : declaration.declaration_byte_end
                ].decode("utf-8"),
                content_indent=declaration.content_indent,
            )
        return context_text

    def get_dump_for_embedding(self, root: Node):
        context = self.get_context(root)
        content = self.get_content(root)
        return """{context}{content}""".format(
            context=context if context else "",
            content=content,
        )

    def __add__(self, other: "TempChunk"):
        return TempChunk(
            nodes=self.nodes + other.nodes,
            parent_declarations=list(
                set(self.parent_declarations).intersection(other.parent_declarations)
            ),
        )


class DocumentParser:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.max_tokens = int(self.embedding_model.get_max_seq_length())  # type: ignore
        self.break_chunks_at = 512

    def _get_str_token_count(self, text: str) -> int:
        return len(self.embedding_model.tokenize([text])["input_ids"][0])

    def _get_node_token_count(self, node: Node, last_end_byte: int, root: Node) -> int:
        return self._get_str_token_count(root.text[last_end_byte : node.end_byte].decode("utf-8"))

    def _get_chunk_tokens(self, chunk: TempChunk, root: Node) -> int:
        return self._get_str_token_count(chunk.get_dump_for_embedding(root))

    def _chunk_nodes_by_whitespace(
        self,
        node: Node,
        language: str,
        parent_declarations: list[ParentDeclaration] = [],
        root_node: Node | None = None,
        last_end_byte=0,
    ) -> list[TempChunk]:
        """
        This function takes a list of nodes and chunks them by start-end points that are touching.
        Touching nodes are those where the end point of one node is adjacent to the start point of the next node.

        Args:
        node (Node): The node to chunk up.
        root_node (Node): The root node of the tree.

        Returns:
        List[List[Node]]: A list of lists, where each sublist contains touching nodes.
        """
        children = node.children
        root_node = root_node or node

        # Initialize the first chunk
        chunks: list[TempChunk] = []
        chunk_token_count = 0

        parent_declarations = parent_declarations.copy()

        def is_touching_last(cur_node: Node):
            spacing_text = root_node.text[last_end_byte : cur_node.start_byte].decode("utf-8")

            return len([c for c in spacing_text if c == "\n"]) < 2

        for i in range(len(children)):
            potential_chunk = TempChunk(
                nodes=[children[i]], parent_declarations=parent_declarations
            )
            token_count = self._get_chunk_tokens(potential_chunk, root_node)

            if token_count > self.break_chunks_at:
                # Recursively chunk the children if the current node is too big or should be chunked
                parent_declarations_for_children = parent_declarations.copy()
                is_parent_declaration = self._node_is_a_declaration(children[i], language)
                if is_parent_declaration:
                    declaration = self._extract_declaration(children[i], root_node, language)
                    if declaration:
                        parent_declarations_for_children.append(declaration)

                children_with_embeddings = self._chunk_nodes_by_whitespace(
                    children[i],
                    language,
                    parent_declarations=parent_declarations_for_children,
                    root_node=root_node,
                    last_end_byte=last_end_byte,
                )

                if len(children_with_embeddings) > 0:
                    # This case is for when the first chunk of the children is touching the last chunk of the current chunks
                    # Usually when the definition of the parent is split from its children
                    # This combines the first logical chunk with its parent definition line.
                    if len(chunks) > 0 and is_touching_last(children_with_embeddings[0].nodes[0]):
                        merged_chunk = chunks[-1] + children_with_embeddings[0]

                        # This forces the merged chunk to be re-embedded
                        chunks[-1] = TempChunk(**merged_chunk.model_dump())
                        if len(children_with_embeddings) > 1:
                            chunks.extend(children_with_embeddings[1:])
                    else:
                        chunks.extend(children_with_embeddings)

                    chunk_token_count = self._get_chunk_tokens(chunks[-1], root_node)
                    last_end_byte = chunks[-1].nodes[-1].end_byte
                continue

            if len(chunks) > 0:
                # The node_token_count doesn't include the parent declaration etc
                node_token_count = self._get_node_token_count(children[i], last_end_byte, root_node)
                if (
                    is_touching_last(children[i])
                    and chunk_token_count + node_token_count < self.max_tokens
                ):
                    # If it touches, add it to the current chunk
                    chunks[-1].nodes.append(children[i])
                    chunk_token_count += node_token_count

                    last_end_byte = children[i].end_byte
                    continue

            chunks.append(potential_chunk)
            chunk_token_count = token_count
            last_end_byte = children[i].end_byte

        for chunk in chunks:
            # Filter out the declarations that are already in the chunk
            chunk.parent_declarations = [
                d
                for d in chunk.parent_declarations
                if not set(d.declaration_nodes).intersection(chunk.nodes)
            ]

        return chunks

    def _node_is_a_declaration(self, node: Node, language: str) -> bool:
        if language == "python":
            return node.type.endswith("_definition") or any(
                [child.type == "block" for child in node.children]
            )  # is a definition type node or has an immediate block child
        if language in ["javascript", "typescript", "jsx", "tsx"]:
            return node.type in [
                "class_declaration",
                "method_definition",
                "function_declaration",
                "lexical_declaration",
            ] or any(
                [child.type == "statement_block" for child in node.children]
            )  # is a definition type node or has an immediate block child
        return False

    def _extract_declaration(
        self, node: Node, root_node: Node, language: str
    ) -> ParentDeclaration | None:
        if language == "python":
            result = index_with_node_type(":", node, recursive=False)
            if result is None:
                # Handle the case where there is no colon
                return None
            index_of_colon, _ = result

            declaration_start_byte = get_indent_start_byte(node.children[0], root_node)
            declaration_end_byte = node.children[index_of_colon].end_byte
            content_indent = root_node.text[
                get_indent_start_byte(node.children[index_of_colon + 1], root_node) : node.children[
                    index_of_colon + 1
                ].start_byte
            ].decode("utf-8")

            return ParentDeclaration(
                id=node.id,
                declaration_byte_start=declaration_start_byte,
                declaration_byte_end=declaration_end_byte,
                content_indent=content_indent,
                declaration_nodes=node.children[: index_of_colon + 1],
            )

        if language in ["javascript", "typescript", "jsx", "tsx"]:
            child = first_child_with_type(set(("class_body", "statement_block")), node)
            if child is None:
                return None

            result = index_with_node_type("{", child, recursive=True)
            if result is None:
                return None
            child_index, bracket_node = result

            if bracket_node.next_sibling:
                declaration_start_byte = get_indent_start_byte(node.children[0], root_node)
                declaration_end_byte = bracket_node.end_byte
                content_indent = root_node.text[
                    get_indent_start_byte(
                        bracket_node.next_sibling, root_node
                    ) : bracket_node.next_sibling.start_byte
                ].decode("utf-8")

                declaration_nodes = node.children[:child_index]
                if bracket_node.parent:
                    # TODO: There probably are more levels to this...
                    bracket_node_index = bracket_node.parent.children.index(bracket_node)
                    declaration_nodes += bracket_node.parent.children[: bracket_node_index + 1]

                return ParentDeclaration(
                    id=node.id,
                    declaration_byte_start=declaration_start_byte,
                    declaration_byte_end=declaration_end_byte,
                    content_indent=content_indent,
                    declaration_nodes=declaration_nodes,
                )
        return None

    @class_method_lru_cache(maxsize=16)
    def _get_parser(self, language: str):
        return tree_sitter_languages.get_parser(language)

    def _chunk_document(self, document: Document) -> list[DocumentChunk]:
        tree = self._get_parser(document.language).parse(bytes(document.text, "utf-8"))

        chunked_documents = self._chunk_nodes_by_whitespace(tree.root_node, document.language)

        chunks: list[DocumentChunk] = []
        last_line = 1

        for i, tmp_chunk in enumerate(chunked_documents):
            context_text = tmp_chunk.get_context(tree.root_node)
            chunk_text = tmp_chunk.get_content(tree.root_node)
            embedding_dump = tmp_chunk.get_dump_for_embedding(tree.root_node)

            chunk = DocumentChunk(
                index=i,
                first_line_number=last_line,
                last_line_number=last_line + len(chunk_text),
                context=context_text,
                content=chunk_text.strip("\n"),
                path=document.path,
                # Hash should be unique to the file, it is used in comparing which chunks changed
                hash=self._generate_sha256_hash(f"[{document.path}][{i}]\n{embedding_dump}"),
                token_count=self._get_str_token_count(embedding_dump),
                repo_id=document.repo_id,
                language=document.language,
            )

            last_line += len(chunk_text.split("\n"))

            chunks.append(chunk)

        return chunks

    def _generate_sha256_hash(self, text: str):
        return hashlib.sha256(text.encode("utf-8"), usedforsecurity=False).hexdigest()

    def process_document(self, document: Document) -> list[DocumentChunk]:
        """
        Process a document by chunking it into smaller pieces and extracting metadata about each chunk.
        """
        start_start = time.time()
        chunks = self._chunk_document(document)
        logger.debug(f"Document chunking took {time.time() - start_start:.2f} seconds")

        return chunks

    def process_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """
        Process a list of documents by chunking them into smaller pieces and extracting metadata about each chunk.
        """
        chunks = []
        for i, document in enumerate(documents):
            chunks.extend(self.process_document(document))
            logger.debug(f"Processed document {i + 1}/{len(documents)}")
        return chunks
