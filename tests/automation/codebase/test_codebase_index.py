import os
import textwrap
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import (
    BaseDocumentChunk,
    CodebaseNamespaceStatus,
    Document,
    EmbeddedDocumentChunk,
)
from seer.automation.codebase.namespace import CodebaseNamespaceManager
from seer.automation.codebase.state import DummyCodebaseStateManager
from seer.automation.codebase.storage_adapters import FilesystemStorageAdapter
from seer.automation.models import (
    EventDetails,
    ExceptionDetails,
    RepoDefinition,
    Stacktrace,
    StacktraceFrame,
)
from seer.db import DbCodebaseNamespace, DbRepositoryInfo, Session


class TestCodebaseIndexCreateAndIndex(unittest.TestCase):
    def setUp(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "filesystem"
        os.environ["CODEBASE_STORAGE_DIR"] = "data/tests/chroma/storage"
        os.environ["CODEBASE_WORKSPACE_DIR"] = "data/tests/chroma/workspaces"

    def tearDown(self) -> None:
        FilesystemStorageAdapter.clear_all_storage()
        return super().tearDown()

    @patch("seer.automation.codebase.codebase_index.RepoClient")
    def test_simple_create(self, mock_repo_client):
        mock_repo_client.from_repo_definition.return_value.get_branch_head_sha = MagicMock(
            return_value="sha"
        )
        namespace_id = CodebaseIndex.create(
            organization=1,
            project=1,
            repo=RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="1"),
            tracking_branch="main",
        )

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
                self.assertEqual(namespace.id, namespace_id)
                self.assertEqual(namespace.tracking_branch, "main")
                self.assertEqual(namespace.sha, "sha")
                self.assertEqual(namespace.status, CodebaseNamespaceStatus.PENDING)

                workspace = CodebaseNamespaceManager.load_workspace(namespace.id, skip_copy=True)
                self.assertIsNotNone(workspace)
                if workspace:
                    self.assertEqual(workspace.namespace.id, namespace.id)

    @patch("seer.automation.codebase.codebase_index.CodebaseIndex.embed_chunks")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.RepoClient")
    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    def test_simple_create_and_index(
        self,
        mock_cleanup_dir,
        mock_repo_client,
        read_specific_files,
        mock_embed_chunks,
    ):
        mock_repo_client.from_repo_definition.return_value.get_branch_head_sha = MagicMock(
            return_value="sha"
        )
        mock_repo_client.from_repo_info.return_value.load_repo_to_tmp_dir.return_value = (
            "tmp_dir",
            "tmp_dir/repo",
        )
        mock_repo_client.from_repo_info.return_value.repo.full_name = "getsentry/seer"

        read_specific_files.return_value = [
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

        namespace_id = CodebaseIndex.create(
            organization=1,
            project=1,
            repo=RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="1"),
            tracking_branch="main",
        )

        codebase_index = CodebaseIndex.index(
            namespace_id=namespace_id,
            embedding_model=MagicMock(),
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
                self.assertEqual(namespace.status, CodebaseNamespaceStatus.CREATED)

                workspace = CodebaseNamespaceManager.load_workspace(namespace.id)
                self.assertIsNotNone(workspace)
                if workspace:
                    self.assertEqual(workspace.namespace.id, namespace.id)

    @patch("seer.automation.codebase.codebase_index.CodebaseIndex.embed_chunks")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.RepoClient")
    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    def test_failing_create_and_index(
        self, mock_cleanup_dir, mock_repo_client, read_specific_files, mock_embed_chunks
    ):
        mock_repo_client.from_repo_definition.return_value.get_branch_head_sha = MagicMock(
            return_value="sha"
        )
        mock_repo_client.from_repo_definition.return_value.load_repo_to_tmp_dir.return_value = (
            "tmp_dir",
            "tmp_dir/repo",
        )
        mock_repo_client.from_repo_definition.return_value.repo.full_name = "getsentry/seer"

        read_specific_files.return_value = [
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
        mock_embed_chunks.side_effect = Exception("Error during embedding")

        namespace_id = CodebaseIndex.create(
            organization=1,
            project=1,
            repo=RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="1"),
            tracking_branch="main",
        )

        with self.assertRaises(Exception) as context:
            CodebaseIndex.index(
                namespace_id=namespace_id,
                embedding_model=MagicMock(),
            )
            self.assertEqual(str(context.exception), "Error during embedding")

        with Session() as session:
            repo_info = session.query(DbRepositoryInfo).first()
            self.assertIsNotNone(repo_info)
            if repo_info:
                self.assertEqual(repo_info.external_slug, "getsentry/seer")
                self.assertEqual(repo_info.provider, "github")
                self.assertEqual(repo_info.organization, 1)
                self.assertEqual(repo_info.project, 1)

            namespace = session.query(DbCodebaseNamespace).first()
            self.assertIsNone(namespace)


class TestCodebaseIndexUpdate(unittest.TestCase):
    def setUp(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "filesystem"
        os.environ["CODEBASE_STORAGE_DIR"] = "data/tests/chroma/storage"
        os.environ["CODEBASE_WORKSPACE_DIR"] = "data/tests/chroma/workspaces"

        self.embedding_model = MagicMock()
        self.embedding_model.encode.return_value = [np.ones((768))]

        self.namespace = CodebaseNamespaceManager.create_namespace_with_new_or_existing_repo(
            1,
            1,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="1"),
            "sha",
            "main",
            should_set_as_default=True,
        )
        self.namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
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
        return super().tearDown()

    def mock_embed_chunks(self, chunks: list[BaseDocumentChunk], embedding_model: Any):
        return [EmbeddedDocumentChunk(**dict(chunk), embedding=np.ones((768))) for chunk in chunks]

    # @patch("seer.automation.codebase.codebase_index.RepoClient.from_repo_info")
    # @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    # def test_update_no_changes(self, mock_cleanup_dir, mock_repo_client_from_repo_info):
    #     print("starting test_update_no_changes")
    #     mock_repo_client = MagicMock()
    #     mock_repo_client_from_repo_info.return_value = mock_repo_client
    #     mock_repo_client.get_commit_file_diffs.return_value = ([], [])

    #     codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
    #     codebase_index.embed_chunks = self.mock_embed_chunks
    #     codebase_index.update()

    #     mock_repo_client.load_repo_to_tmp_dir.assert_not_called()
    #     self.assertEqual(codebase_index.workspace.namespace.sha, "sha")
    #     print("complete test_update_no_changes")

    # @patch("seer.automation.codebase.codebase_index.RepoClient.from_repo_info")
    # @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    # @patch("seer.automation.codebase.codebase_index.read_specific_files")
    # @patch("seer.automation.codebase.codebase_index.DocumentParser")
    # def test_update_with_simple_chunk_add(
    #     self,
    #     mock_document_parser,
    #     mock_read_specific_files,
    #     mock_cleanup_dir,
    #     mock_repo_client_from_repo_info,
    # ):
    #     print("starting test_update_with_simple_chunk_add")
    #     mock_repo_client = MagicMock()
    #     mock_repo_client_from_repo_info.return_value = mock_repo_client
    #     mock_repo_client.load_repo_to_tmp_dir.return_value = (
    #         "tmp_dir",
    #         "tmp_dir/repo",
    #     )
    #     mock_repo_client.get_branch_head_sha.return_value = "new_sha"
    #     mock_repo_client.get_commit_file_diffs.return_value = (["file1.py"], [])

    #     mock_read_specific_files.return_value = {"file1.py": "content"}
    #     mock_document_parser.return_value.process_documents.return_value = [
    #         BaseDocumentChunk(
    #             context="context",
    #             index=0,
    #             path="file1.py",
    #             hash="file1new",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #         BaseDocumentChunk(
    #             context="def foo():",
    #             content="return x",
    #             language="python",
    #             path="file1.py",
    #             hash="file1.1",
    #             token_count=1,
    #             index=1,
    #         ),
    #     ]

    #     codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
    #     codebase_index.embed_chunks = self.mock_embed_chunks

    #     codebase_index.update()

    #     mock_read_specific_files.assert_called_once()
    #     mock_document_parser.return_value.process_documents.assert_called_once()
    #     mock_cleanup_dir.assert_called_once()

    #     with Session() as session:
    #         db_namespace = session.get(DbCodebaseNamespace, 1)
    #         self.assertIsNotNone(db_namespace)
    #         if db_namespace:
    #             self.assertEqual(db_namespace.sha, "new_sha")

    #     workspace = CodebaseNamespaceManager.load_workspace(1)

    #     self.assertIsNotNone(workspace)
    #     if workspace:
    #         chunks = workspace.get_chunks_for_paths(["file1.py"])
    #         chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
    #         self.assertEqual(chunk_hashes, ["file1new", "file1.1"])

    #     print("complete test_update_with_simple_chunk_add")

    # @patch("seer.automation.codebase.codebase_index.RepoClient.from_repo_info")
    # @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    # @patch("seer.automation.codebase.codebase_index.read_specific_files")
    # @patch("seer.automation.codebase.codebase_index.DocumentParser")
    # def test_update_with_chunk_addition(
    #     self,
    #     mock_document_parser,
    #     mock_read_specific_files,
    #     mock_cleanup_dir,
    #     mock_repo_client_from_repo_info,
    # ):
    #     print("starting test_update_with_chunk_addition")
    #     mock_repo_client = MagicMock()
    #     mock_repo_client_from_repo_info.return_value = mock_repo_client
    #     mock_repo_client.load_repo_to_tmp_dir.return_value = (
    #         "tmp_dir",
    #         "tmp_dir/repo",
    #     )
    #     mock_repo_client.get_branch_head_sha.return_value = "new_sha"
    #     mock_repo_client.get_commit_file_diffs.return_value = (["file1.py"], [])

    #     mock_read_specific_files.return_value = {"file1.py": "content"}
    #     mock_document_parser.return_value.process_documents.return_value = [
    #         BaseDocumentChunk(
    #             context="context",
    #             index=0,
    #             path="file1.py",
    #             hash="file1",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #         BaseDocumentChunk(
    #             context="context",
    #             index=1,
    #             path="file1.py",
    #             hash="file1.1new",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #         BaseDocumentChunk(
    #             context="context",
    #             index=2,
    #             path="file1.py",
    #             hash="file1.2new",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #     ]

    #     codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
    #     codebase_index.embed_chunks = self.mock_embed_chunks

    #     codebase_index.update()

    #     mock_read_specific_files.assert_called_once()
    #     mock_document_parser.return_value.process_documents.assert_called_once()
    #     mock_cleanup_dir.assert_called_once()

    #     with Session() as session:
    #         db_namespace = session.get(DbCodebaseNamespace, 1)
    #         self.assertIsNotNone(db_namespace)
    #         if db_namespace:
    #             self.assertEqual(db_namespace.sha, "new_sha")

    #     workspace = CodebaseNamespaceManager.load_workspace(1)

    #     self.assertIsNotNone(workspace)
    #     if workspace:
    #         chunks = workspace.get_chunks_for_paths(["file1.py"])
    #         chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
    #         self.assertEqual(chunk_hashes, ["file1", "file1.1new", "file1.2new"])

    #     print("complete test_update_with_chunk_addition")

    # @patch("seer.automation.codebase.codebase_index.RepoClient.from_repo_info")
    # @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    # @patch("seer.automation.codebase.codebase_index.read_specific_files")
    # @patch("seer.automation.codebase.codebase_index.DocumentParser")
    # def test_update_with_complete_chunk_replacement(
    #     self,
    #     mock_document_parser,
    #     mock_read_specific_files,
    #     mock_cleanup_dir,
    #     mock_repo_client_from_repo_info,
    # ):
    #     mock_repo_client = MagicMock()
    #     mock_repo_client_from_repo_info.return_value = mock_repo_client
    #     mock_repo_client.load_repo_to_tmp_dir.return_value = (
    #         "tmp_dir",
    #         "tmp_dir/repo",
    #     )
    #     mock_repo_client.get_branch_head_sha.return_value = "new_sha"
    #     mock_repo_client.get_commit_file_diffs.return_value = (["file1.py"], [])
    #     mock_read_specific_files.return_value = {"file1.py": "content"}
    #     mock_document_parser.return_value.process_documents.return_value = [
    #         BaseDocumentChunk(
    #             context="context",
    #             index=0,
    #             path="file1.py",
    #             hash="file1new",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #         BaseDocumentChunk(
    #             context="context",
    #             index=1,
    #             path="file1.py",
    #             hash="file1.1new",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #         BaseDocumentChunk(
    #             context="context",
    #             index=2,
    #             path="file1.py",
    #             hash="file1.2new",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #     ]

    #     codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
    #     codebase_index.embed_chunks = self.mock_embed_chunks

    #     codebase_index.update()

    #     mock_read_specific_files.assert_called_once()
    #     mock_document_parser.return_value.process_documents.assert_called_once()
    #     mock_cleanup_dir.assert_called_once()

    #     with Session() as session:
    #         db_namespace = session.get(DbCodebaseNamespace, 1)
    #         self.assertIsNotNone(db_namespace)
    #         if db_namespace:
    #             self.assertEqual(db_namespace.sha, "new_sha")

    #     workspace = CodebaseNamespaceManager.load_workspace(1)

    #     self.assertIsNotNone(workspace)
    #     if workspace:
    #         chunks = workspace.get_chunks_for_paths(["file1.py"])
    #         chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
    #         self.assertEqual(chunk_hashes, ["file1new", "file1.1new", "file1.2new"])

    #     print("complete test_update_with_complete_chunk_replacement")

    # @patch("seer.automation.codebase.codebase_index.RepoClient.from_repo_info")
    # @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    # @patch("seer.automation.codebase.codebase_index.read_specific_files")
    # @patch("seer.automation.codebase.codebase_index.DocumentParser")
    # def test_update_with_index_change(
    #     self,
    #     mock_document_parser,
    #     mock_read_specific_files,
    #     mock_cleanup_dir,
    #     mock_repo_client_from_repo_info,
    # ):
    #     mock_repo_client = MagicMock()
    #     mock_repo_client_from_repo_info.return_value = mock_repo_client
    #     mock_repo_client.load_repo_to_tmp_dir.return_value = (
    #         "tmp_dir",
    #         "tmp_dir/repo",
    #     )
    #     mock_repo_client.get_branch_head_sha.return_value = "new_sha"
    #     mock_repo_client.get_commit_file_diffs.return_value = (["file1.py"], [])
    #     mock_read_specific_files.return_value = {"file1.py": "content"}
    #     mock_document_parser.return_value.process_documents.return_value = [
    #         BaseDocumentChunk(
    #             context="context",
    #             index=0,
    #             path="file1.py",
    #             hash="file1",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #         BaseDocumentChunk(
    #             context="context",
    #             index=1,
    #             path="file1.py",
    #             hash="file1.0.1new",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #         BaseDocumentChunk(
    #             context="context",
    #             index=2,
    #             path="file1.py",
    #             hash="file1.1",
    #             language="python",
    #             token_count=1,
    #             content="content",
    #         ),
    #     ]

    #     codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
    #     codebase_index.embed_chunks = self.mock_embed_chunks

    #     codebase_index.update()

    #     mock_read_specific_files.assert_called_once()
    #     mock_document_parser.return_value.process_documents.assert_called_once()
    #     mock_cleanup_dir.assert_called_once()

    #     with Session() as session:
    #         db_namespace = session.get(DbCodebaseNamespace, 1)
    #         self.assertIsNotNone(db_namespace)
    #         if db_namespace:
    #             self.assertEqual(db_namespace.sha, "new_sha")

    #     workspace = CodebaseNamespaceManager.load_workspace(1)

    #     self.assertIsNotNone(workspace)
    #     if workspace:
    #         chunks = workspace.get_chunks_for_paths(["file1.py"])
    #         chunk_hashes = [chunk.hash for chunk in sorted(chunks, key=lambda x: x.index)]
    #         self.assertEqual(chunk_hashes, ["file1", "file1.0.1new", "file1.1"])

    # @patch("seer.automation.codebase.codebase_index.RepoClient.mock_repo_client")
    # @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    # @patch("seer.automation.codebase.codebase_index.read_specific_files")
    # @patch("seer.automation.codebase.codebase_index.DocumentParser")
    # def test_update_with_full_delete(
    #     self,
    #     mock_document_parser,
    #     mock_read_specific_files,
    #     mock_cleanup_dir,
    #     mock_repo_client_from_repo_info,
    # ):
    #     mock_repo_client = MagicMock()
    #     mock_repo_client_from_repo_info.return_value = mock_repo_client
    #     mock_repo_client.load_repo_to_tmp_dir.return_value = (
    #         "tmp_dir",
    #         "tmp_dir/repo",
    #     )
    #     mock_repo_client.get_branch_head_sha.return_value = "new_sha"
    #     mock_repo_client.get_commit_file_diffs.return_value = (["file1.py"], [])
    #     mock_read_specific_files.return_value = {"file1.py": "content"}
    #     mock_document_parser.return_value.process_documents.return_value = []

    #     codebase_index = CodebaseIndex.from_repo_id(1, embedding_model=self.embedding_model)
    #     codebase_index.embed_chunks = self.mock_embed_chunks

    #     codebase_index.update()

    #     mock_read_specific_files.assert_called_once()
    #     mock_document_parser.return_value.process_documents.assert_called_once()
    #     mock_cleanup_dir.assert_called_once()

    #     with Session() as session:
    #         db_namespace = session.get(DbCodebaseNamespace, 1)
    #         self.assertIsNotNone(db_namespace)
    #         if db_namespace:
    #             self.assertEqual(db_namespace.sha, "new_sha")

    #     workspace = CodebaseNamespaceManager.load_workspace(1)

    #     self.assertIsNotNone(workspace)
    #     if workspace:
    #         chunks = workspace.get_chunks_for_paths(["file1.py"])
    #         self.assertEqual(chunks, [])


class TestCodebaseIndexDiffContainsStacktraceFiles(unittest.TestCase):
    def setUp(self):
        self.organization = 1
        self.project = 1
        self.repo_definition = MagicMock()
        self.repo_client = MagicMock()
        self.repo_client.load_repo_to_tmp_dir.return_value = ("tmp_dir", "tmp_dir/repo")

        self.namespace = CodebaseNamespaceManager.create_namespace_with_new_or_existing_repo(
            1,
            1,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="1"),
            "sha",
            "main",
        )
        self.state_manager = DummyCodebaseStateManager()

        self.codebase_index = CodebaseIndex(
            self.organization,
            self.project,
            self.repo_client,
            self.namespace,
            state_manager=self.state_manager,
            embedding_model=MagicMock(),
        )

    def test_diff_contains_stacktrace_files_with_intersection(self):
        # Mock the get_commit_file_diffs method to return changed and removed files
        self.repo_client.get_commit_file_diffs.return_value = (
            ["file1.py", "file2.py"],
            ["file3.py"],
        )
        # Create a stacktrace with one of the files that has changed
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file2.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file2.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files
        self.assertTrue(self.codebase_index.diff_contains_stacktrace_files(event_details))

    def test_diff_contains_stacktrace_files_without_intersection(self):
        # Mock the get_commit_file_diffs method to return changed and removed files
        self.repo_client.get_commit_file_diffs.return_value = (
            ["file1.py", "file2.py"],
            ["file3.py"],
        )
        # Create a stacktrace with files that have not changed
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file4.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file4.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files
        self.assertFalse(self.codebase_index.diff_contains_stacktrace_files(event_details))

    def test_diff_contains_stacktrace_files_with_removed_file(self):
        # Mock the get_commit_file_diffs method to return changed and removed files
        self.repo_client.get_commit_file_diffs.return_value = (
            ["file1.py"],
            ["file2.py"],
        )
        # Create a stacktrace with a file that has been removed
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file2.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file2.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files
        self.assertTrue(self.codebase_index.diff_contains_stacktrace_files(event_details))

    def test_diff_contains_stacktrace_files_raises_file_not_found(self):
        # Mock the get_commit_file_diffs method to raise FileNotFoundError
        self.repo_client.get_commit_file_diffs.side_effect = FileNotFoundError
        # Create a stacktrace with any file
        stacktrace = Stacktrace(
            frames=[
                StacktraceFrame(
                    filename="file1.py",
                    col_no=0,
                    line_no=10,
                    function="test",
                    context=[],
                    abs_path="file1.py",
                )
            ]
        )
        event_details = EventDetails(
            title="yes",
            exceptions=[ExceptionDetails(type="yes", value="yes", stacktrace=stacktrace)],
        )
        # Check if the diff contains stacktrace files raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.codebase_index.diff_contains_stacktrace_files(event_details)


class TestCodebaseIndexFileIntegrityCheck(unittest.TestCase):
    def setUp(self):
        self.mock_repo_client = MagicMock()
        self.namespace = CodebaseNamespaceManager.create_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
        )
        self.codebase = CodebaseIndex(
            1, 1, self.mock_repo_client, self.namespace, MagicMock(), MagicMock()
        )

    def test_integrity_check_success(self):
        self.namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1.py",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                ),
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk2hash",
                    path="path2.js",
                    index=0,
                    token_count=0,
                    language="javascript",
                    embedding=np.ones((768)),
                ),
            ]
        )

        self.mock_repo_client.get_index_file_set.return_value = set(["path1.py", "path2.js"])

        assert self.codebase.verify_file_integrity() is True

    def test_integrity_check_fail_missing_file(self):
        self.namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1.py",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                )
            ]
        )

        self.mock_repo_client.get_index_file_set.return_value = set(["path1.py", "path2.js"])

        assert self.codebase.verify_file_integrity() is False

    def test_integrity_check_fail_extra_file(self):
        self.namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1.py",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                ),
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk2hash",
                    path="path2.js",
                    index=0,
                    token_count=0,
                    language="javascript",
                    embedding=np.ones((768)),
                ),
            ]
        )

        self.mock_repo_client.get_index_file_set.return_value = set(["path1.py"])

        assert self.codebase.verify_file_integrity() is False

    def test_integrity_check_ignores_unsupported_exts(self):
        self.namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1.py",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                ),
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk2hash",
                    path="path2.js",
                    index=0,
                    token_count=0,
                    language="javascript",
                    embedding=np.ones((768)),
                ),
            ]
        )

        self.mock_repo_client.get_index_file_set.return_value = set(
            ["path1.py", "path2.js", "unsupported.ext", "bad.no", ".gitignore"]
        )

        assert self.codebase.verify_file_integrity() is False
