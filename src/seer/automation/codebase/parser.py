import hashlib
import logging
import textwrap
import time

import tree_sitter_languages
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tree_sitter import Node
from tqdm import tqdm

from seer.automation.codebase.ast import (
    AstDeclaration,
    extract_declaration,
    first_child_with_type,
    get_indent_start_byte,
    index_with_node_type,
    node_is_a_declaration,
)
from seer.automation.codebase.models import BaseDocumentChunk, Document
from seer.utils import class_method_lru_cache

logger = logging.getLogger("autofix")


class TempChunk(BaseModel):
    nodes: list[Node]
    parent_declarations: list[AstDeclaration]

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
                declaration=declaration.to_str(root, include_indent=True),
                content_indent=root.text[
                    declaration.indent_start_byte : declaration.declaration_start_byte
                ].decode("utf-8"),
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
        self.max_tokens = int(self.embedding_model.get_max_seq_length())
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
        parent_declarations: list[AstDeclaration] = [],
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
                is_parent_declaration = node_is_a_declaration(children[i], language)
                if is_parent_declaration:
                    declaration = extract_declaration(children[i], root_node, language)
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

    @class_method_lru_cache(maxsize=16)
    def _get_parser(self, language: str):
        return tree_sitter_languages.get_parser(language)

    def _chunk_document(self, document: Document) -> list[BaseDocumentChunk]:
        tree = self._get_parser(document.language).parse(bytes(document.text, "utf-8"))

        chunked_documents = self._chunk_nodes_by_whitespace(tree.root_node, document.language)

        chunks: list[BaseDocumentChunk] = []

        for i, tmp_chunk in enumerate(chunked_documents):
            context_text = tmp_chunk.get_context(tree.root_node)
            chunk_text = tmp_chunk.get_content(tree.root_node)
            embedding_dump = tmp_chunk.get_dump_for_embedding(tree.root_node)

            chunk = BaseDocumentChunk(
                index=i,
                context=context_text,
                content=chunk_text.strip("\n"),
                path=document.path,
                # Hash should be unique to the file, it is used in comparing which chunks changed
                hash=self._generate_sha256_hash(f"[{document.path}][{i}]\n{embedding_dump}"),
                token_count=self._get_str_token_count(embedding_dump),
                language=document.language,
            )

            chunks.append(chunk)

        return chunks

    def _generate_sha256_hash(self, text: str):
        return hashlib.sha256(text.encode("utf-8"), usedforsecurity=False).hexdigest()

    def process_document(self, document: Document) -> list[BaseDocumentChunk]:
        """
        Process a document by chunking it into smaller pieces and extracting metadata about each chunk.
        """
        return self._chunk_document(document)

    def process_documents(self, documents: list[Document]) -> list[BaseDocumentChunk]:
        """
        Process a list of documents by chunking them into smaller pieces and extracting metadata about each chunk.
        """
        chunks = []

        for i, document in tqdm(enumerate(documents), total=len(documents)):
            chunks.extend(self.process_document(document))
        return chunks
