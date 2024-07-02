import difflib

import sentry_sdk
import tree_sitter_languages
from tree_sitter import Tree
from unidiff import PatchSet

from seer.automation.codebase.ast import (
    extract_declaration,
    find_first_parent_declaration,
    supports_parent_declarations,
)
from seer.automation.codebase.models import BaseDocument
from seer.automation.codebase.utils import get_language_from_path
from seer.automation.models import FileChange, FilePatch, Hunk, Line


def copy_document_and_apply_changes(
    document: BaseDocument, file_changes: list[FileChange]
) -> BaseDocument | None:
    content: str | None = document.text
    # Make sure the changes are applied in order!
    changes = list(filter(lambda x: x.path == document.path, file_changes))
    if changes:
        for change in changes:
            content = change.apply(content)

    if content is None or content == "":
        return None

    return BaseDocument(path=document.path, text=content)


def make_file_patches(
    file_changes: list[FileChange],
    document_paths: list[str],
    original_documents: list[BaseDocument],
) -> tuple[list[FilePatch], str]:
    changed_documents_map: dict[str, BaseDocument] = {}

    diffs: list[str] = []
    for i, document in enumerate(original_documents):
        if document and document.text:
            changed_document = copy_document_and_apply_changes(document, file_changes)

            diff = difflib.unified_diff(
                document.text.splitlines(),
                changed_document.text.splitlines() if changed_document else "",
                fromfile=document.path,
                tofile=changed_document.path if changed_document else "/dev/null",
                lineterm="",
            )

            diff_str = "\n".join(diff).strip("\n")
            diffs.append(diff_str)

            if changed_document:
                changed_documents_map[changed_document.path] = changed_document
        else:
            path = document_paths[i]
            changed_document = copy_document_and_apply_changes(
                BaseDocument(path=path, text=""), file_changes
            )

            if changed_document:
                diff = difflib.unified_diff(
                    [],  # Empty list to represent no original content
                    changed_document.text.splitlines(),
                    fromfile="/dev/null",
                    tofile=path,
                    lineterm="",
                )

                diff_str = "\n".join(diff).strip("\n")
                diffs.append(diff_str)
                changed_documents_map[path] = changed_document

    file_patches = []
    for file_diff in diffs:
        patches = PatchSet(file_diff)
        if not patches:
            sentry_sdk.capture_message(f"No patches for diff: {file_diff}")
            continue
        patched_file = patches[0]

        tree: Tree | None = None
        doc = changed_documents_map.get(patched_file.path)
        language = get_language_from_path(patched_file.path)
        if doc and language and supports_parent_declarations(language):
            ast_parser = tree_sitter_languages.get_parser(language)
            tree = ast_parser.parse(doc.text.encode("utf-8"))

        hunks: list[Hunk] = []
        for hunk in patched_file:
            lines: list[Line] = []
            for line in hunk:
                lines.append(
                    Line(
                        source_line_no=line.source_line_no,
                        target_line_no=line.target_line_no,
                        diff_line_no=line.diff_line_no,
                        value=line.value,
                        line_type=line.line_type,
                    )
                )

            section_header = hunk.section_header
            if tree and doc:
                line_numbers = [
                    line.target_line_no
                    for line in lines
                    if line.line_type != " " and line.target_line_no is not None
                ]
                first_line_no = line_numbers[0] if line_numbers else None
                last_line_no = line_numbers[-1] if line_numbers else None
                if first_line_no is not None and last_line_no is not None:
                    node = tree.root_node.descendant_for_point_range(
                        (first_line_no, 0), (last_line_no, 0)
                    )
                    if node and language:
                        parent_declaration_node = find_first_parent_declaration(node, language)
                        declaration = (
                            extract_declaration(parent_declaration_node, tree.root_node, language)
                            if parent_declaration_node
                            else None
                        )
                        section_header_str = (
                            declaration.to_str(tree.root_node, include_indent=False)
                            if declaration
                            else ""
                        )
                        if section_header_str:
                            section_header_lines = section_header_str.splitlines()
                            if section_header_lines:
                                section_header = section_header_lines[0].strip()

            hunks.append(
                Hunk(
                    source_start=hunk.source_start,
                    source_length=hunk.source_length,
                    target_start=hunk.target_start,
                    target_length=hunk.target_length,
                    section_header=section_header,
                    lines=lines,
                )
            )
        patch_type = (
            patched_file.is_added_file and "A" or patched_file.is_removed_file and "D" or "M"
        )
        file_patches.append(
            FilePatch(
                type=patch_type,
                path=patched_file.path,
                added=patched_file.added,
                removed=patched_file.removed,
                source_file=patched_file.source_file,
                target_file=patched_file.target_file,
                hunks=hunks,
            )
        )

    combined_diff = "\n".join(diffs)

    return file_patches, combined_diff
