from pydantic import BaseModel
from tree_sitter import Node, Tree


class AstDeclaration(BaseModel):
    id: int
    indent_start_byte: int
    declaration_start_byte: int
    declaration_end_byte: int

    declaration_nodes: list[Node]

    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other):
        if isinstance(other, AstDeclaration):
            return self.id == other.id
        return False

    def __hash__(self):
        return self.id

    def to_str(self, root_node: Node, include_indent=False) -> str:
        return root_node.text[
            (
                self.indent_start_byte if include_indent else self.declaration_start_byte
            ) : self.declaration_end_byte
        ].decode("utf-8")


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


def supports_parent_declarations(language: str) -> bool:
    return language in ["python", "javascript", "typescript", "jsx", "tsx"]


def node_is_a_declaration(node: Node, language: str) -> bool:
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


def find_first_parent_declaration(node: Node, language: str):
    parent = node.parent
    while parent:
        if node_is_a_declaration(parent, language):
            return parent


def extract_declaration(node: Node, root_node: Node, language: str) -> AstDeclaration | None:
    if language == "python":
        result = index_with_node_type(":", node, recursive=False)
        if result is None:
            # Handle the case where there is no colon
            return None
        index_of_colon, _ = result

        indent_start_byte = get_indent_start_byte(node.children[0], root_node)
        declaration_start_byte = node.children[0].start_byte
        declaration_end_byte = node.children[index_of_colon].end_byte

        return AstDeclaration(
            id=node.id,
            indent_start_byte=indent_start_byte,
            declaration_start_byte=declaration_start_byte,
            declaration_end_byte=declaration_end_byte,
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
            indent_start_byte = get_indent_start_byte(node.children[0], root_node)
            declaration_start_byte = bracket_node.next_sibling.start_byte
            declaration_end_byte = bracket_node.end_byte

            declaration_nodes = node.children[:child_index]
            if bracket_node.parent:
                # TODO: There probably are more levels to this...
                bracket_node_index = bracket_node.parent.children.index(bracket_node)
                declaration_nodes += bracket_node.parent.children[: bracket_node_index + 1]

            return AstDeclaration(
                id=node.id,
                indent_start_byte=indent_start_byte,
                declaration_start_byte=declaration_start_byte,
                declaration_end_byte=declaration_end_byte,
                declaration_nodes=declaration_nodes,
            )
    return None
