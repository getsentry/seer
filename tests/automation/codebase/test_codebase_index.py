import datetime
import os
import textwrap
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import BaseDocumentChunk, Document, EmbeddedDocumentChunk
from seer.automation.codebase.namespace import CodebaseNamespaceManager
from seer.automation.codebase.storage_adapters import FilesystemStorageAdapter
from seer.automation.models import FileChange, RepoDefinition
from seer.db import DbCodebaseNamespace, DbRepositoryInfo, Session


class TestCodebaseIndexCreate(unittest.TestCase):
    def setUp(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "filesystem"
        os.environ["CODEBASE_STORAGE_DIR"] = "data/tests/chroma/storage"
        os.environ["CODEBASE_WORKSPACE_DIR"] = "data/tests/chroma/workspaces"

        self.repo_client = MagicMock()
        self.repo_client.load_repo_to_tmp_dir.return_value = ("tmp_dir", "tmp_dir/repo")
        self.repo_client.repo.full_name = "getsentry/seer"

    def tearDown(self) -> None:
        FilesystemStorageAdapter.clear_all_storage()
        FilesystemStorageAdapter.clear_all_workspaces()
        return super().tearDown()

    @patch("seer.automation.codebase.codebase_index.CodebaseIndex.embed_chunks")
    @patch("seer.automation.codebase.codebase_index.read_directory")
    @patch("seer.automation.codebase.codebase_index.RepoClient")
    def test_simple_create(self, mock_repo_client, mock_read_directory, mock_embed_chunks):
        mock_repo_client.return_value = self.repo_client
        self.repo_client.get_branch_head_sha = MagicMock(return_value="sha")

        mock_read_directory.return_value = [
            Document(
                path="file1.py",
                language="python",
                text=textwrap.dedent(
                    """\
                    def foo():
                        x = 1 + 1
                        y = 2 + 2
                        return x + y
                    """
                ),
            )
        ]
        mock_embed_chunks.return_value = [
            EmbeddedDocumentChunk(
                id=1,
                context="def foo():",
                content="x = 1 + 1",
                language="python",
                path="file1.py",
                hash="file1",
                token_count=1,
                index=0,
                embedding=np.ones((768)),
            )
        ]

        codebase_index = CodebaseIndex.create(
            organization=1,
            project=1,
            repo=RepoDefinition(provider="github", owner="getsentry", name="seer"),
            embedding_model=MagicMock(),
            tracking_branch="main",
        )

        self.assertEqual(codebase_index.namespace.tracking_branch, "main")

        with Session() as session:
            repo_info = session.query(DbRepositoryInfo).first()
            self.assertIsNotNone(repo_info)
            if repo_info:
                self.assertEqual(repo_info.external_slug, "getsentry/seer")
                self.assertEqual(repo_info.provider, "github")
                self.assertEqual(repo_info.organization, 1)
                self.assertEqual(repo_info.project, 1)

            namespace = session.query(DbCodebaseNamespace).first()
            self.assertIsNotNone(namespace)
            if namespace:
                self.assertEqual(namespace.tracking_branch, "main")
                self.assertEqual(namespace.sha, "sha")

                workspace = CodebaseNamespaceManager.load_workspace(namespace.id)
                self.assertIsNotNone(workspace)
                if workspace:
                    self.assertEqual(workspace.namespace.id, namespace.id)


class TestCodebaseIndexUpdate(unittest.TestCase):
    def setUp(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "filesystem"
        os.environ["CODEBASE_STORAGE_DIR"] = "data/tests/chroma/storage"
        os.environ["CODEBASE_WORKSPACE_DIR"] = "data/tests/chroma/workspaces"

        self.repo_client = MagicMock()
        self.repo_client.load_repo_to_tmp_dir.return_value = ("tmp_dir", "tmp_dir/repo")
        self.repo_client.repo.full_name = "getsentry/seer"

        self.embedding_model = MagicMock()
        self.embedding_model.encode.return_value = [np.ones((768))]

        self.namespace = CodebaseNamespaceManager.create_namespace_with_new_or_existing_repo(
            1, 1, "github", "getsentry/seer", "sha", "main"
        )
        self.namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    id=1,
                    context="def foo():",
                    content="x = 1 + 1",
                    language="python",
                    path="file1.py",
                    hash="file1",
                    token_count=1,
                    index=0,
                    embedding=np.ones((768)),
                ),
                EmbeddedDocumentChunk(
                    id=1,
                    context="def foo():",
                    content="return x",
                    language="python",
                    path="file1.py",
                    hash="file1.1",
                    token_count=1,
                    index=1,
                    embedding=np.ones((768)),
                ),
            ]
        )
        self.namespace.save()

    def tearDown(self) -> None:
        FilesystemStorageAdapter.clear_all_storage()
        FilesystemStorageAdapter.clear_all_workspaces()
        return super().tearDown()

    def mock_embed_chunks(self, chunks: list[BaseDocumentChunk], embedding_model: Any):
        return [EmbeddedDocumentChunk(**dict(chunk), embedding=np.ones((768))) for chunk in chunks]

    def test_update_no_changes(self):
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=([], []))

        codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
        codebase_index.repo_client = self.repo_client
        codebase_index.update()

        self.repo_client.load_repo_to_tmp_dir.assert_not_called()
        self.assertEqual(codebase_index.workspace.namespace.sha, "sha")

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    @patch("seer.automation.codebase.namespace.datetime.datetime", wraps=datetime.datetime)
    def test_update_with_simple_chunk_add(
        self, mock_dt, mock_document_parser, mock_read_specific_files, mock_cleanup_dir
    ):
        self.repo_client.get_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents.return_value = [
            BaseDocumentChunk(
                id=1,
                context="context",
                index=0,
                path="file1.py",
                hash="file1new",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                id=1,
                context="def foo():",
                content="return x",
                language="python",
                path="file1.py",
                hash="file1.1",
                token_count=1,
                index=1,
            ),
        ]

        mock_date = datetime.datetime(2022, 1, 1, 12, 0)
        mock_dt.now.return_value = mock_date

        codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
        codebase_index.repo_client = self.repo_client
        codebase_index.embed_chunks = self.mock_embed_chunks

        codebase_index.update()

        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()

        with Session() as session:
            db_namespace = session.get(DbCodebaseNamespace, 1)
            self.assertIsNotNone(db_namespace)
            if db_namespace:
                self.assertEqual(db_namespace.sha, "new_sha")
                self.assertEqual(db_namespace.accessed_at, mock_date)
                self.assertEqual(db_namespace.updated_at, mock_date)

        workspace = CodebaseNamespaceManager.load_workspace(1)

        self.assertIsNotNone(workspace)
        if workspace:
            chunks = workspace.get_chunks_for_paths(["file1.py"])
            chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
            self.assertEqual(chunk_hashes, ["file1new", "file1.1"])

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    @patch("seer.automation.codebase.namespace.datetime.datetime", wraps=datetime.datetime)
    def test_update_with_chunk_addition(
        self, mock_dt, mock_document_parser, mock_read_specific_files, mock_cleanup_dir
    ):
        self.repo_client.get_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents.return_value = [
            BaseDocumentChunk(
                id=1,
                context="context",
                index=0,
                path="file1.py",
                hash="file1",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                id=2,
                context="context",
                index=1,
                path="file1.py",
                hash="file1.1new",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                id=3,
                context="context",
                index=2,
                path="file1.py",
                hash="file1.2new",
                language="python",
                token_count=1,
                content="content",
            ),
        ]

        mock_date = datetime.datetime(2022, 1, 1, 12, 0)
        mock_dt.now.return_value = mock_date

        codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
        codebase_index.repo_client = self.repo_client
        codebase_index.embed_chunks = self.mock_embed_chunks

        codebase_index.update()

        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()

        with Session() as session:
            db_namespace = session.get(DbCodebaseNamespace, 1)
            self.assertIsNotNone(db_namespace)
            if db_namespace:
                self.assertEqual(db_namespace.sha, "new_sha")
                self.assertEqual(db_namespace.accessed_at, mock_date)
                self.assertEqual(db_namespace.updated_at, mock_date)

        workspace = CodebaseNamespaceManager.load_workspace(1)

        self.assertIsNotNone(workspace)
        if workspace:
            chunks = workspace.get_chunks_for_paths(["file1.py"])
            chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
            self.assertEqual(chunk_hashes, ["file1", "file1.1new", "file1.2new"])

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    @patch("seer.automation.codebase.namespace.datetime.datetime", wraps=datetime.datetime)
    def test_update_with_complete_chunk_replacement(
        self, mock_dt, mock_document_parser, mock_read_specific_files, mock_cleanup_dir
    ):
        self.repo_client.get_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents.return_value = [
            BaseDocumentChunk(
                id=1,
                context="context",
                index=0,
                path="file1.py",
                hash="file1new",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                id=2,
                context="context",
                index=1,
                path="file1.py",
                hash="file1.1new",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                id=3,
                context="context",
                index=2,
                path="file1.py",
                hash="file1.2new",
                language="python",
                token_count=1,
                content="content",
            ),
        ]

        mock_date = datetime.datetime(2022, 1, 1, 12, 0)
        mock_dt.now.return_value = mock_date

        codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
        codebase_index.repo_client = self.repo_client
        codebase_index.embed_chunks = self.mock_embed_chunks

        codebase_index.update()

        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()

        with Session() as session:
            db_namespace = session.get(DbCodebaseNamespace, 1)
            self.assertIsNotNone(db_namespace)
            if db_namespace:
                self.assertEqual(db_namespace.sha, "new_sha")
                self.assertEqual(db_namespace.accessed_at, mock_date)
                self.assertEqual(db_namespace.updated_at, mock_date)

        workspace = CodebaseNamespaceManager.load_workspace(1)

        self.assertIsNotNone(workspace)
        if workspace:
            chunks = workspace.get_chunks_for_paths(["file1.py"])
            chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
            self.assertEqual(chunk_hashes, ["file1new", "file1.1new", "file1.2new"])

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    @patch("seer.automation.codebase.namespace.datetime.datetime", wraps=datetime.datetime)
    def test_update_with_index_change(
        self, mock_dt, mock_document_parser, mock_read_specific_files, mock_cleanup_dir
    ):
        self.repo_client.get_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents.return_value = [
            BaseDocumentChunk(
                id=1,
                context="context",
                index=0,
                path="file1.py",
                hash="file1",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                id=4,
                context="context",
                index=1,
                path="file1.py",
                hash="file1.0.1new",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                id=2,
                context="context",
                index=2,
                path="file1.py",
                hash="file1.1",
                language="python",
                token_count=1,
                content="content",
            ),
        ]

        mock_date = datetime.datetime(2022, 1, 1, 12, 0)
        mock_dt.now.return_value = mock_date

        codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
        codebase_index.repo_client = self.repo_client
        codebase_index.embed_chunks = self.mock_embed_chunks

        codebase_index.update()

        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()

        with Session() as session:
            db_namespace = session.get(DbCodebaseNamespace, 1)
            self.assertIsNotNone(db_namespace)
            if db_namespace:
                self.assertEqual(db_namespace.sha, "new_sha")
                self.assertEqual(db_namespace.accessed_at, mock_date)
                self.assertEqual(db_namespace.updated_at, mock_date)

        workspace = CodebaseNamespaceManager.load_workspace(1)

        self.assertIsNotNone(workspace)
        if workspace:
            chunks = workspace.get_chunks_for_paths(["file1.py"])
            chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
            self.assertEqual(chunk_hashes, ["file1", "file1.0.1new", "file1.1"])

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    @patch("seer.automation.codebase.namespace.datetime.datetime", wraps=datetime.datetime)
    def test_update_with_full_delete(
        self, mock_dt, mock_document_parser, mock_read_specific_files, mock_cleanup_dir
    ):
        self.repo_client.get_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents.return_value = []

        mock_date = datetime.datetime(2022, 1, 1, 12, 0)
        mock_dt.now.return_value = mock_date

        codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
        codebase_index.repo_client = self.repo_client
        codebase_index.embed_chunks = self.mock_embed_chunks

        codebase_index.update()

        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()

        with Session() as session:
            db_namespace = session.get(DbCodebaseNamespace, 1)
            self.assertIsNotNone(db_namespace)
            if db_namespace:
                self.assertEqual(db_namespace.sha, "new_sha")
                self.assertEqual(db_namespace.accessed_at, mock_date)
                self.assertEqual(db_namespace.updated_at, mock_date)

        workspace = CodebaseNamespaceManager.load_workspace(1)

        self.assertIsNotNone(workspace)
        if workspace:
            chunks = workspace.get_chunks_for_paths(["file1.py"])
            self.assertEqual(chunks, [])


class TestCodebaseIndexGetFilePatches(unittest.TestCase):
    def setUp(self):
        self.organization = 1
        self.project = 1
        self.repo_definition = MagicMock()
        self.repo_client = MagicMock()
        self.repo_client.load_repo_to_tmp_dir.return_value = ("tmp_dir", "tmp_dir/repo")

        self.namespace = CodebaseNamespaceManager.create_namespace_with_new_or_existing_repo(
            1, 1, "github", "getsentry/seer", "sha", "main"
        )

        self.codebase_index = CodebaseIndex(
            self.organization,
            self.project,
            self.repo_client,
            self.namespace,
            embedding_model=MagicMock(),
        )
        self.mock_get_document = MagicMock()
        self.codebase_index.get_document = self.mock_get_document

    def tearDown(self) -> None:
        FilesystemStorageAdapter.clear_all_storage()
        FilesystemStorageAdapter.clear_all_workspaces()
        return super().tearDown()

    def test_get_file_patches(self):
        def get_document_mock(file_path, ignore_local_changes=False):
            if file_path == "file1.py":
                return Document(
                    path="file1.py",
                    language="python",
                    text=textwrap.dedent(
                        """\
                        def foo():
                            x = 1 + 1
                            y = 2 + 2
                            return x + y"""
                    ),
                )
            elif file_path == "file2.py":
                return None
            elif file_path == "file3.py":
                return Document(
                    path="file3.py",
                    language="python",
                    text=textwrap.dedent(
                        """\
                        def baz():
                            a = 1 + 5
                            return 'baz'
                        """
                    ),
                )

        self.mock_get_document.side_effect = get_document_mock

        self.codebase_index.file_changes = [
            FileChange(
                change_type="edit",
                path="file1.py",
                reference_snippet="""    y = 2 + 2""",
                new_snippet="""    y = 2 + 3""",
            ),
            FileChange(
                change_type="edit",
                path="file1.py",
                reference_snippet="""    y = 2 + 3""",
                new_snippet="""    y = 2 + 4 # yes\n    z = 3 + 3""",
            ),
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
            FileChange(
                change_type="delete", path="file3.py", reference_snippet="""    a = 1 + 5\n"""
            ),
        ]

        # Execute
        patches, diff_str = self.codebase_index.get_file_patches()
        patches.sort(key=lambda p: p.path)  # Sort patches by path to make the test deterministic

        # Assert
        self.assertEqual(len(patches), 3)
        self.assertEqual(patches[0].path, "file1.py")
        self.assertEqual(patches[0].type, "M")
        self.assertEqual(patches[0].added, 2)
        self.assertEqual(patches[0].removed, 1)
        self.assertEqual(patches[0].source_file, "file1.py")
        self.assertEqual(patches[0].target_file, "file1.py")
        self.assertEqual(len(patches[0].hunks), 1)

        self.assertEqual(patches[1].path, "file2.py")
        self.assertEqual(patches[1].type, "A")
        self.assertEqual(patches[1].added, 2)
        self.assertEqual(patches[1].removed, 0)
        self.assertEqual(patches[1].source_file, "/dev/null")
        self.assertEqual(patches[1].target_file, "file2.py")
        self.assertEqual(len(patches[1].hunks), 1)

        self.assertEqual(patches[2].path, "file3.py")
        self.assertEqual(patches[2].type, "M")
        self.assertEqual(patches[2].added, 0)
        self.assertEqual(patches[2].removed, 1)
        self.assertEqual(patches[2].source_file, "file3.py")
        self.assertEqual(patches[2].target_file, "file3.py")
        self.assertEqual(len(patches[2].hunks), 1)

    def test_get_file_patches_with_full_delete(self):
        self.mock_get_document.return_value = Document(
            path="file_to_delete.py",
            language="python",
            text=textwrap.dedent(
                """\
                def baz():
                    a = 1 + 5
                    return 'baz'
                """
            ),
        )
        file_change = FileChange(
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
        )
        self.codebase_index.file_changes = [file_change]

        # Execute
        patches, diff_str = self.codebase_index.get_file_patches()

        # Assert
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].path, "file_to_delete.py")
        self.assertEqual(patches[0].type, "D")
        self.assertEqual(patches[0].added, 0)
        self.assertEqual(patches[0].removed, 3)
        self.assertEqual(patches[0].source_file, "file_to_delete.py")
        self.assertEqual(patches[0].target_file, "/dev/null")

    def test_section_header(self):
        self.mock_get_document.return_value = Document(
            path="file1.py",
            language="python",
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

        self.codebase_index.file_changes = [
            FileChange(
                change_type="edit",
                path="file1.py",
                reference_snippet="""    y = 2 + 2""",
                new_snippet="""    y = 2 + 3""",
            ),
            FileChange(
                change_type="edit",
                path="file1.py",
                reference_snippet="""    y = 2 + 3""",
                new_snippet="""    y = 2 + 4 # yes\n        z = 3 + 3""",
            ),
        ]

        # Execute
        patches, diff_str = self.codebase_index.get_file_patches()
        patches.sort(key=lambda p: p.path)  # Sort patches by path to make the test deterministic

        print(patches)

        # Assert
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].path, "file1.py")
        self.assertEqual(patches[0].type, "M")
        self.assertEqual(len(patches[0].hunks), 1)
        self.assertEqual(patches[0].hunks[0].section_header, "def foobar(a: int):")
