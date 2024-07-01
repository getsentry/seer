import textwrap
import unittest

from seer.automation.codebase.file_patches import make_file_patches
from seer.automation.codebase.models import BaseDocument
from seer.automation.models import FileChange


class TestMakeFilePatches(unittest.TestCase):
    def test_make_file_patches_edit(self):
        file_changes = [
            FileChange(
                change_type="edit",
                path="file1.py",
                reference_snippet="    y = 2 + 2",
                new_snippet="    y = 2 + 3",
            ),
            FileChange(
                change_type="edit",
                path="file1.py",
                reference_snippet="    y = 2 + 3",
                new_snippet="    y = 2 + 4 # yes\n    z = 3 + 3",
            ),
        ]
        document_paths = ["file1.py"]
        original_documents = [
            BaseDocument(
                path="file1.py",
                text=textwrap.dedent(
                    """\
                def foo():
                    x = 1 + 1
                    y = 2 + 2
                    return x + y"""
                ),
            )
        ]

        patches, diff_str = make_file_patches(file_changes, document_paths, original_documents)

        assert len(patches) == 1
        assert patches[0].path == "file1.py"
        assert patches[0].type == "M"
        assert patches[0].added == 2
        assert patches[0].removed == 1
        assert patches[0].source_file == "file1.py"
        assert patches[0].target_file == "file1.py"
        assert len(patches[0].hunks) == 1
        assert patches[0].hunks[0].section_header == "def foo():"

    def test_make_file_patches_create(self):
        file_changes = [
            FileChange(
                change_type="create",
                path="file2.py",
                new_snippet=textwrap.dedent(
                    """\
                def bar():
                    return 'bar'
                """
                ),
            ),
        ]
        document_paths = ["file2.py"]
        original_documents = [None]

        patches, diff_str = make_file_patches(file_changes, document_paths, original_documents)  # type: ignore

        assert len(patches) == 1
        assert patches[0].path == "file2.py"
        assert patches[0].type == "A"
        assert patches[0].added == 2
        assert patches[0].removed == 0
        assert patches[0].source_file == "/dev/null"
        assert patches[0].target_file == "file2.py"
        assert len(patches[0].hunks) == 1

    def test_make_file_patches_delete(self):
        file_changes = [
            FileChange(
                change_type="delete",
                path="file3.py",
                reference_snippet="    a = 1 + 5\n",
            ),
        ]
        document_paths = ["file3.py"]
        original_documents = [
            BaseDocument(
                path="file3.py",
                text=textwrap.dedent(
                    """\
                def baz():
                    a = 1 + 5
                    return 'baz'
                """
                ),
            )
        ]

        patches, diff_str = make_file_patches(file_changes, document_paths, original_documents)

        assert len(patches) == 1
        assert patches[0].path == "file3.py"
        assert patches[0].type == "M"
        assert patches[0].added == 0
        assert patches[0].removed == 1
        assert patches[0].source_file == "file3.py"
        assert patches[0].target_file == "file3.py"

    def test_make_file_patches_full_delete(self):
        file_changes = [
            FileChange(
                change_type="delete",
                path="file_to_delete.py",
                reference_snippet=textwrap.dedent(
                    """\
                def baz():
                    a = 1 + 5
                    return 'baz'
                """
                ),
                new_snippet=None,
            ),
        ]
        document_paths = ["file_to_delete.py"]
        original_documents = [
            BaseDocument(
                path="file_to_delete.py",
                text=textwrap.dedent(
                    """\
                def baz():
                    a = 1 + 5
                    return 'baz'
                """
                ),
            )
        ]

        patches, diff_str = make_file_patches(file_changes, document_paths, original_documents)

        assert len(patches) == 1
        assert patches[0].path == "file_to_delete.py"
        assert patches[0].type == "D"
        assert patches[0].added == 0
        assert patches[0].removed == 3
        assert patches[0].source_file == "file_to_delete.py"
        assert patches[0].target_file == "/dev/null"

    def test_make_file_patches_with_section_header(self):
        file_changes = [
            FileChange(
                change_type="edit",
                path="file1.py",
                reference_snippet="    y = 2 + 2",
                new_snippet="    y = 2 + 3",
            ),
        ]
        document_paths = ["file1.py"]
        original_documents = [
            BaseDocument(
                path="file1.py",
                text=textwrap.dedent(
                    """\
                class foo:
                    a = 5
                    def foobar(a: int):
                        x = 1 + 1
                        y = 2 + 2
                        return x + y"""
                ),
            )
        ]

        patches, diff_str = make_file_patches(file_changes, document_paths, original_documents)

        assert len(patches) == 1
        assert patches[0].path == "file1.py"
        assert patches[0].type == "M"
        assert len(patches[0].hunks) == 1
        assert patches[0].hunks[0].section_header == "def foobar(a: int):"
