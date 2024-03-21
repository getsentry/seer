from typing import Any, Literal

from johen.pytest import parametrize
from pydantic import BaseModel

from seer.automation.codebase.ast import find_first_parent_declaration


class FakeNode(BaseModel):
    type: Literal[
        "class_declaration",
        "method_definition",
        "function_declaration",
        "lexical_declaration",
        "something_else",
        "class_definition",
        "method_definition",
        "function_definition",
        "code",
        "blah",
        "foo",
    ]
    parent: Any = None
    children: list[Any] = []


@parametrize
def test_find_first_parent_declaration_js_declaration(
    root: FakeNode,
    parents: list[FakeNode],
    language: Literal["javascript", "typescript", "jsx", "tsx"],
):
    valid_declaration_types = [
        "class_declaration",
        "method_definition",
        "function_declaration",
        "lexical_declaration",
    ]
    child = root
    first_parent = None
    for parent in parents:
        if first_parent is None and parent.type in valid_declaration_types:
            first_parent = parent
        root.parent = parent
        root = parent
    parent = find_first_parent_declaration(child, language, max_depth=len(parents))  # type: ignore
    assert parent == first_parent


@parametrize
def test_find_first_parent_declaration_py_declaration(
    root: FakeNode,
    parents: list[FakeNode],
    language: Literal["python"],
):
    valid_declaration_types = ["class_definition", "method_definition", "function_definition"]
    child = root
    first_parent = None
    for parent in parents:
        if first_parent is None and parent.type in valid_declaration_types:
            first_parent = parent
        root.parent = parent
        root = parent
    parent = find_first_parent_declaration(child, language, max_depth=len(parents))  # type: ignore
    assert parent == first_parent


@parametrize
def test_max_depth(
    root: FakeNode,
    parents: list[FakeNode],
    language: Literal["python"],
):
    valid_declaration_types = ["class_definition", "method_definition", "function_definition"]
    child = root
    first_parent = None
    for i, parent in enumerate(parents):
        if i == 0 and first_parent is None and parent.type in valid_declaration_types:
            first_parent = parent
        root.parent = parent
        root = parent
    parent = find_first_parent_declaration(child, language, max_depth=1)  # type: ignore
    assert parent == first_parent
