import unittest
import uuid
from unittest.mock import MagicMock, patch

import numpy as np

from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import BaseDocumentChunk, EmbeddedDocumentChunk, RepositoryInfo
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
            self.organization, self.project, self.repo_client, self.repo_info, self.run_id
        )

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
    @patch("seer.automation.codebase.codebase_index.get_embedding_model")
    def test_update_with_simple_chunk_add(
        self,
        mock_get_embedding_model,
        mock_document_parser,
        mock_read_specific_files,
        mock_cleanup_dir,
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_embedding_model = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding_model
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

        self.codebase_index._embed_chunks = MagicMock()
        self.codebase_index._embed_chunks.return_value = [
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
    @patch("seer.automation.codebase.codebase_index.get_embedding_model")
    def test_update_with_chunk_replacement(
        self,
        mock_get_embedding_model,
        mock_document_parser,
        mock_read_specific_files,
        mock_cleanup_dir,
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=(["file1.py"], []))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_embedding_model = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding_model
        mock_document_parser.return_value.process_documents = MagicMock()
        mock_document_parser.return_value.process_documents.return_value = [
            BaseDocumentChunk(
                id=2,
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
                        id=1,
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
                        id=2,
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

        self.codebase_index._embed_chunks = MagicMock()
        self.codebase_index._embed_chunks.return_value = [
            EmbeddedDocumentChunk(
                id=3,
                context="context",
                index=1,
                path="file1.py",
                hash="file1.1.1",
                language="python",
                token_count=1,
                content="content",
                embedding=np.ones((768)),
            ),
            EmbeddedDocumentChunk(
                id=4,
                context="context",
                index=2,
                path="file1.py",
                hash="file1.2.1",
                language="python",
                token_count=1,
                content="content",
                embedding=np.ones((768)),
            ),
        ]

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
    @patch("seer.automation.codebase.codebase_index.get_embedding_model")
    def test_update_with_removed_file(
        self,
        mock_get_embedding_model,
        mock_document_parser,
        mock_read_specific_files,
        mock_cleanup_dir,
    ):
        # Setup
        self.repo_client.get_default_branch_head_sha = MagicMock(return_value="new_sha")
        self.repo_client.get_commit_file_diffs = MagicMock(return_value=([], ["file2.py"]))
        mock_read_specific_files.return_value = {"file1.py": "content"}
        mock_embedding_model = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding_model
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
                        id=2,
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
                        id=3,
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

        self.codebase_index._embed_chunks = MagicMock()
        self.codebase_index._embed_chunks.return_value = []

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
