import textwrap
import unittest
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import (
    BaseDocumentChunk,
    Document,
    EmbeddedDocumentChunk,
    RepositoryInfo,
)
from seer.automation.models import FileChange, FilePatch, Hunk, Line
from seer.db import DbDocumentChunk, DbRepositoryInfo, Session


class TestCodebaseIndexUpdate(unittest.TestCase):
    def setUp(self):
        self.organization = 1
        self.project = 1
        self.repo_definition = MagicMock()
        self.run_id = uuid.uuid4()
        self.repo_info = RepositoryInfo(
            id=1, organization=1, project=1, provider="github", external_slug="test/repo", sha="sha"
        )
        self.repo_client = MagicMock()
        self.repo_client.load_repo_to_tmp_dir.return_value = ("tmp_dir", "tmp_dir/repo")
        self.codebase_index = CodebaseIndex(
            self.organization,
            self.project,
            self.repo_client,
            self.repo_info,
            self.run_id,
            embedding_model=MagicMock(),
        )

    def mock_embed_chunks(self, chunks: list[BaseDocumentChunk], embedding_model: Any):
        return [EmbeddedDocumentChunk(**dict(chunk), embedding=np.ones((768))) for chunk in chunks]

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.Session")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    def test_update_no_changes(
        self, mock_document_parser, mock_read_specific_files, mock_session, mock_cleanup_dir
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=([], []))

        # Execute
        self.codebase_index.update()

        # Assert
        self.repo_client.load_repo_to_tmp_dir.assert_not_called()
        mock_read_specific_files.assert_not_called()
        mock_document_parser.assert_not_called()
        mock_session.assert_not_called()
        mock_cleanup_dir.assert_not_called()
        self.assertEqual(self.codebase_index.repo_info.sha, "sha")

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    def test_update_with_simple_chunk_add(
        self,
        mock_document_parser,
        mock_read_specific_files,
        mock_cleanup_dir,
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents = MagicMock()

        with Session() as session:
            session.add(
                DbRepositoryInfo(
                    id=1,
                    organization=1,
                    project=1,
                    provider="github",
                    external_slug="test/repo",
                    sha="sha",
                )
            )
            session.commit()

        self.codebase_index.embed_chunks = MagicMock()
        self.codebase_index.embed_chunks.return_value = [
            EmbeddedDocumentChunk(
                id=1,
                context="context",
                index=1,
                path="file1.py",
                hash="file1",
                language="python",
                token_count=1,
                content="content",
                embedding=np.ones((768)),
            )
        ]

        # Execute
        self.codebase_index.update()

        # Assert
        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()
        self.assertEqual(self.codebase_index.repo_info.sha, "new_sha")

        with Session() as session:
            updated_repo_info = session.get(DbRepositoryInfo, 1)
            self.assertIsNotNone(updated_repo_info)
            if updated_repo_info:
                self.assertEqual(updated_repo_info.sha, "new_sha")
                added_chunks = (
                    session.query(DbDocumentChunk).filter(DbDocumentChunk.hash == "file1").all()
                )
                self.assertEqual(len(added_chunks), 1)
                self.assertEqual(added_chunks[0].hash, "file1")

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    def test_update_with_chunk_replacement(
        self,
        mock_document_parser,
        mock_read_specific_files,
        mock_cleanup_dir,
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents = MagicMock()
        mock_document_parser.return_value.process_documents.return_value = [
            BaseDocumentChunk(
                context="context",
                index=1,
                path="file1.py",
                hash="file1.1.1",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                context="context",
                index=2,
                path="file1.py",
                hash="file1.2.1",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                context="context",
                index=3,
                path="file1.py",
                hash="file1.2",
                language="python",
                token_count=1,
                content="content",
            ),
        ]

        with Session() as session:
            session.add(
                DbRepositoryInfo(
                    id=1,
                    organization=1,
                    project=1,
                    provider="github",
                    external_slug="test/repo",
                    sha="sha",
                )
            )

            session.flush()
            session.add_all(
                [
                    EmbeddedDocumentChunk(
                        context="context",
                        index=1,
                        path="file1.py",
                        hash="file1.1",
                        language="python",
                        token_count=1,
                        content="content",
                        embedding=np.ones((768)),
                    ).to_db_model(1),
                    EmbeddedDocumentChunk(
                        context="context",
                        index=2,
                        path="file1.py",
                        hash="file1.2",
                        language="python",
                        token_count=1,
                        content="content",
                        embedding=np.ones((768)),
                    ).to_db_model(1),
                ]
            )

            session.commit()

        self.codebase_index.embed_chunks = MagicMock()
        self.codebase_index.embed_chunks.side_effect = self.mock_embed_chunks

        # Execute
        self.codebase_index.update()

        # Assert
        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()
        self.assertEqual(self.codebase_index.repo_info.sha, "new_sha")

        with Session() as session:
            chunks = (
                session.query(DbDocumentChunk)
                .where(DbDocumentChunk.path == "file1.py")
                .order_by("index")
                .all()
            )
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].hash, "file1.1.1")
            self.assertEqual(chunks[1].hash, "file1.2.1")
            self.assertEqual(chunks[2].hash, "file1.2")
            self.assertEqual(chunks[2].index, 3)

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    def test_update_with_removed_file(
        self,
        mock_document_parser,
        mock_read_specific_files,
        mock_cleanup_dir,
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=([], ["file2.py"]))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents = MagicMock()
        mock_document_parser.return_value.process_documents.return_value = []

        with Session() as session:
            session.add(
                DbRepositoryInfo(
                    id=1,
                    organization=1,
                    project=1,
                    provider="github",
                    external_slug="test/repo",
                    sha="sha",
                )
            )

            session.flush()
            session.add_all(
                [
                    EmbeddedDocumentChunk(
                        context="context",
                        index=2,
                        path="file1.py",
                        hash="file1.1",
                        language="python",
                        token_count=1,
                        content="content",
                        embedding=np.ones((768)),
                    ).to_db_model(1),
                    EmbeddedDocumentChunk(
                        context="context",
                        index=2,
                        path="file2.py",
                        hash="file2.1",
                        language="python",
                        token_count=1,
                        content="content",
                        embedding=np.ones((768)),
                    ).to_db_model(1),
                ]
            )

            session.commit()

        self.codebase_index.embed_chunks = MagicMock()
        self.codebase_index.embed_chunks.side_effect = self.mock_embed_chunks

        # Execute
        self.codebase_index.update()

        # Assert
        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()
        self.assertEqual(self.codebase_index.repo_info.sha, "new_sha")

        with Session() as session:
            chunks = session.query(DbDocumentChunk).order_by("index").all()
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].hash, "file1.1")

    @patch("seer.automation.codebase.codebase_index.cleanup_dir")
    @patch("seer.automation.codebase.codebase_index.read_specific_files")
    @patch("seer.automation.codebase.codebase_index.DocumentParser")
    def test_update_with_temporary_chunk_replacement(
        self,
        mock_document_parser,
        mock_read_specific_files,
        mock_cleanup_dir,
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_document_parser.return_value.process_documents = MagicMock()
        mock_document_parser.return_value.process_documents.return_value = [
            BaseDocumentChunk(
                context="context",
                index=1,
                path="file1.py",
                hash="file1.1.1",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                context="context",
                index=2,
                path="file1.py",
                hash="file1.2.1",
                language="python",
                token_count=1,
                content="content",
            ),
            BaseDocumentChunk(
                context="context",
                index=3,
                path="file1.py",
                hash="file1.2",
                language="python",
                token_count=1,
                content="content",
            ),
        ]

        with Session() as session:
            session.add(
                DbRepositoryInfo(
                    id=1,
                    organization=1,
                    project=1,
                    provider="github",
                    external_slug="test/repo",
                    sha="sha",
                )
            )

            session.flush()
            session.add_all(
                [
                    EmbeddedDocumentChunk(
                        context="context",
                        index=1,
                        path="file1.py",
                        hash="file1.1",
                        language="python",
                        token_count=1,
                        content="content",
                        embedding=np.ones((768)),
                    ).to_db_model(1),
                    EmbeddedDocumentChunk(
                        context="context",
                        index=2,
                        path="file1.py",
                        hash="file1.2",
                        language="python",
                        token_count=1,
                        content="content",
                        embedding=np.ones((768)),
                    ).to_db_model(1),
                ]
            )

            session.commit()

        self.codebase_index.embed_chunks = MagicMock()
        self.codebase_index.embed_chunks.side_effect = self.mock_embed_chunks

        # Execute
        self.codebase_index.update(is_temporary=True)

        # Assert
        mock_read_specific_files.assert_called_once()
        mock_document_parser.return_value.process_documents.assert_called_once()
        mock_cleanup_dir.assert_called_once()
        self.assertEqual(self.codebase_index.repo_info.sha, "new_sha")

        with Session() as session:
            db_repo_info = session.get(DbRepositoryInfo, 1)
            self.assertIsNotNone(db_repo_info)
            if db_repo_info:
                self.assertEqual(db_repo_info.sha, "sha")

            chunks = (
                session.query(DbDocumentChunk)
                .where(DbDocumentChunk.path == "file1.py")
                .where(DbDocumentChunk.namespace is None)
                .order_by("index")
                .all()
            )
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].hash, "file1.1")
            self.assertEqual(chunks[1].hash, "file1.2")
            self.assertEqual(chunks[1].index, 2)

            namespaced_chunks = (
                session.query(DbDocumentChunk)
                .where(DbDocumentChunk.path == "file1.py")
                .where(DbDocumentChunk.namespace == str(self.run_id))
                .order_by("index")
                .all()
            )

            self.assertEqual(len(namespaced_chunks), 3)
            self.assertEqual(namespaced_chunks[0].hash, "file1.1.1")
            self.assertEqual(namespaced_chunks[1].hash, "file1.2.1")
            self.assertEqual(namespaced_chunks[2].hash, "file1.2")
            self.assertEqual(namespaced_chunks[2].index, 3)


class TestCodebaseIndexGetFilePatches(unittest.TestCase):
    def setUp(self):
        self.organization = 1
        self.project = 1
        self.repo_definition = MagicMock()
        self.run_id = uuid.uuid4()
        self.repo_info = RepositoryInfo(
            id=1, organization=1, project=1, provider="github", external_slug="test/repo", sha="sha"
        )
        self.repo_client = MagicMock()
        self.repo_client.load_repo_to_tmp_dir.return_value = ("tmp_dir", "tmp_dir/repo")
        self.codebase_index = CodebaseIndex(
            self.organization,
            self.project,
            self.repo_client,
            self.repo_info,
            self.run_id,
            embedding_model=MagicMock(),
        )
        self.mock_get_document = MagicMock()
        self.codebase_index.get_document = self.mock_get_document

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
