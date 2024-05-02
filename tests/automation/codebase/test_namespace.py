import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from seer.automation.codebase.models import CodebaseNamespace, EmbeddedDocumentChunk
from seer.automation.codebase.namespace import CodebaseNamespaceManager
from seer.automation.codebase.storage_adapters import FilesystemStorageAdapter
from seer.automation.models import RepoDefinition
from seer.db import DbCodebaseNamespace, DbRepositoryInfo, Session


class TestNamespaceManager(unittest.TestCase):
    def setUp(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "filesystem"
        os.environ["CODEBASE_STORAGE_DIR"] = "data/tests/chroma/storage"
        os.environ["CODEBASE_WORKSPACE_DIR"] = "data/tests/chroma/workspaces"

    def tearDown(self) -> None:
        FilesystemStorageAdapter.clear_all_storage()
        return super().tearDown()

    def test_create_repo(self):
        namespace = CodebaseNamespaceManager.create_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
            should_set_as_default=True,
        )
        namespace.save()

        with Session() as session:
            db_repo_info = session.query(DbRepositoryInfo).first()

            self.assertIsNotNone(db_repo_info)
            if db_repo_info:
                self.assertEqual(db_repo_info.id, namespace.repo_info.id)
                self.assertEqual(db_repo_info.external_slug, "getsentry/seer")
                self.assertEqual(db_repo_info.organization, 1)
                self.assertEqual(db_repo_info.project, 1337)

                db_namespace = session.query(DbCodebaseNamespace).first()
                self.assertIsNotNone(db_namespace)
                if db_namespace:
                    self.assertEqual(db_namespace.id, db_repo_info.default_namespace)
                    self.assertEqual(db_namespace.repo_id, db_repo_info.id)
                    self.assertEqual(db_namespace.tracking_branch, "main")
                    self.assertEqual(db_namespace.sha, "sha")

                    namespace = CodebaseNamespace.from_db(db_namespace)
                    storage_location_path = FilesystemStorageAdapter.get_storage_location(
                        db_repo_info.id, namespace.slug
                    )

                    self.assertTrue(os.path.exists(storage_location_path))

    @patch("seer.automation.codebase.namespace.CodebaseNamespaceManager.create_repo")
    @patch(
        "seer.automation.codebase.namespace.CodebaseNamespaceManager.create_or_get_namespace_for_repo"
    )
    def test_get_or_create_namespace_for_repo_existing(
        self, mock_create_namespace_for_repo, mock_create_repo
    ):
        with Session() as session:
            db_repo_info = DbRepositoryInfo(
                organization=1,
                project=1337,
                external_slug="getsentry/seer",
                external_id="123",
                provider="github",
            )
            session.add(db_repo_info)
            session.commit()

        CodebaseNamespaceManager.create_repo = MagicMock()
        CodebaseNamespaceManager.create_or_get_namespace_for_repo = MagicMock()

        CodebaseNamespaceManager.create_repo.assert_not_called()
        CodebaseNamespaceManager.create_namespace_with_new_or_existing_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
        )

        CodebaseNamespaceManager.create_or_get_namespace_for_repo.assert_called_once()

    @patch("seer.automation.codebase.namespace.CodebaseNamespaceManager.create_repo")
    @patch(
        "seer.automation.codebase.namespace.CodebaseNamespaceManager.create_or_get_namespace_for_repo"
    )
    def test_get_or_create_namespace_for_repo_new(
        self, mock_create_namespace_for_repo, mock_create_repo
    ):
        CodebaseNamespaceManager.create_namespace_with_new_or_existing_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
        )

        CodebaseNamespaceManager.create_or_get_namespace_for_repo.assert_not_called()
        CodebaseNamespaceManager.create_repo.assert_called_once()

    def test_insert_chunks(self):
        namespace = CodebaseNamespaceManager.create_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
            should_set_as_default=True,
        )

        namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                ),
            ]
        )

        chunks = namespace.get_chunks_for_paths(["path1"])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].hash, "chunk1hash")

    def test_query_chunks(self):
        namespace = CodebaseNamespaceManager.create_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
            should_set_as_default=True,
        )

        namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)) * -0.1,
                ),
                EmbeddedDocumentChunk(
                    context="chunk2context",
                    content="chunk2",
                    hash="chunk2hash",
                    path="path1",
                    index=2,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)) * 0.99,
                ),
            ]
        )

        chunks = namespace.query_chunks(np.ones((768)) * 0.99, top_k=2)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].hash, "chunk2hash")
        self.assertEqual(chunks[1].hash, "chunk1hash")

    def test_save(self):
        namespace = CodebaseNamespaceManager.create_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
            should_set_as_default=True,
        )

        namespace.save()

        with Session() as session:
            db_namespace = session.query(DbCodebaseNamespace).first()

            self.assertIsNotNone(db_namespace)
            if db_namespace:
                self.assertEqual(db_namespace.tracking_branch, "main")
                self.assertEqual(db_namespace.sha, "sha")

        namespace.namespace.sha = "newsha"
        namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                )
            ]
        )
        namespace.save()
        namespace_id = namespace.namespace.id
        del namespace

        loaded_namespace = CodebaseNamespaceManager.load_workspace(namespace_id)

        self.assertIsNotNone(loaded_namespace)
        if loaded_namespace:
            chunks = loaded_namespace.get_chunks_for_paths(["path1"])

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].hash, "chunk1hash")
            self.assertEqual(loaded_namespace.namespace.sha, "newsha")

    def test_chunk_hashes_exist(self):
        namespace = CodebaseNamespaceManager.create_repo(
            1,
            1337,
            RepoDefinition(provider="github", owner="getsentry", name="seer", external_id="123"),
            "sha",
            tracking_branch="main",
        )

        namespace.insert_chunks(
            [
                EmbeddedDocumentChunk(
                    context="chunk1context",
                    content="chunk1",
                    hash="chunk1hash",
                    path="path1",
                    index=0,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                ),
                EmbeddedDocumentChunk(
                    context="chunk2context",
                    content="chunk2",
                    hash="chunk2hash",
                    path="path1",
                    index=2,
                    token_count=0,
                    language="python",
                    embedding=np.ones((768)),
                ),
            ]
        )

        chunk_hashes = namespace.chunk_hashes_exist(["chunk1hash", "chunk2hash"])

        self.assertEqual(len(chunk_hashes), 2)
        self.assertTrue("chunk1hash" in chunk_hashes)
        self.assertTrue("chunk2hash" in chunk_hashes)

    def test_set_mutex(self):
        repo_id = 1
        namespace_id = 1
        with Session() as session:
            db_repo_info = DbRepositoryInfo(
                id=repo_id,
                organization=1,
                project=1337,
                external_slug="getsentry/seer",
                external_id="123",
                provider="github",
            )
            session.add(db_repo_info)
            session.flush()
            db_namespace = DbCodebaseNamespace(id=namespace_id, repo_id=1, sha="sha")
            session.add(db_namespace)
            session.commit()

        CodebaseNamespaceManager._set_mutex(namespace_id)
        mutex_time = CodebaseNamespaceManager.get_mutex(namespace_id)
        self.assertIsNotNone(mutex_time)

    def test_clear_mutex(self):
        repo_id = 1
        namespace_id = 1
        with Session() as session:
            db_repo_info = DbRepositoryInfo(
                id=repo_id,
                organization=1,
                project=1337,
                external_slug="getsentry/seer",
                external_id="123",
                provider="github",
            )
            session.add(db_repo_info)
            session.flush()
            db_namespace = DbCodebaseNamespace(id=namespace_id, repo_id=1, sha="sha")
            session.add(db_namespace)
            session.commit()

        CodebaseNamespaceManager._set_mutex(namespace_id)
        self.assertIsNotNone(CodebaseNamespaceManager.get_mutex(namespace_id))
        CodebaseNamespaceManager._clear_mutex(namespace_id)
        self.assertIsNone(CodebaseNamespaceManager.get_mutex(namespace_id))

    def test_wait_for_mutex_clear(self):
        repo_id = 1
        namespace_id = 1
        with Session() as session:
            db_repo_info = DbRepositoryInfo(
                id=repo_id,
                organization=1,
                project=1337,
                external_slug="getsentry/seer",
                external_id="123",
                provider="github",
            )
            session.add(db_repo_info)
            session.flush()
            db_namespace = DbCodebaseNamespace(id=namespace_id, repo_id=1, sha="sha")
            session.add(db_namespace)
            session.commit()

        def set_mutex():
            CodebaseNamespaceManager._set_mutex(namespace_id)
            self.assertIsNotNone(CodebaseNamespaceManager.get_mutex(namespace_id))
            time.sleep(1)
            self.assertIsNotNone(CodebaseNamespaceManager.get_mutex(namespace_id))
            CodebaseNamespaceManager._clear_mutex(namespace_id)
            self.assertIsNone(CodebaseNamespaceManager.get_mutex(namespace_id))

        def wait_and_check_mutex():
            self.assertIsNotNone(CodebaseNamespaceManager.get_mutex(namespace_id))
            CodebaseNamespaceManager._wait_for_mutex_clear(namespace_id)
            self.assertIsNone(CodebaseNamespaceManager.get_mutex(namespace_id))

        thread1 = threading.Thread(target=set_mutex)
        thread2 = threading.Thread(target=wait_and_check_mutex)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
