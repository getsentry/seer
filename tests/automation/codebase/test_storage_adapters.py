import os
import unittest
from unittest.mock import MagicMock, create_autospec, patch

from seer.automation.codebase.storage_adapters import (
    FilesystemStorageAdapter,
    GcsStorageAdapter,
    get_storage_adapter_class,
)


class TestStorageAdapter(unittest.TestCase):
    def test_fs_workspace_clear(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "filesystem"
        os.environ["CODEBASE_STORAGE_DIR"] = "data/tests/chroma/storage"
        os.environ["CODEBASE_WORKSPACE_DIR"] = "data/tests/chroma/workspaces"

        StorageAdapter = get_storage_adapter_class()
        adapter = StorageAdapter(1, "test")

        # Create a file in the workspace
        workspace_dir = adapter.tmpdir
        os.makedirs(workspace_dir, exist_ok=True)
        with open(os.path.join(workspace_dir, "test.txt"), "w") as f:
            f.write("test")

        self.assertTrue(os.path.exists(os.path.join(workspace_dir, "test.txt")))
        adapter.clear_workspace()

        self.assertFalse(os.path.exists(workspace_dir))
        self.assertFalse(os.path.exists(os.path.join(workspace_dir, "test.txt")))

    def test_gcs_workspace_clear(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "gcs"
        os.environ["CODEBASE_GCS_BUCKET"] = "seer-chroma"
        os.environ["CODEBASE_GCS_STORAGE_DIR"] = "chroma-test/data/storage"

        StorageAdapter = get_storage_adapter_class()
        adapter = StorageAdapter(1, "test")

        # Create a file in the workspace
        workspace_dir = adapter.tmpdir
        os.makedirs(workspace_dir, exist_ok=True)
        with open(os.path.join(workspace_dir, "test.txt"), "w") as f:
            f.write("test")

        self.assertTrue(os.path.exists(os.path.join(workspace_dir, "test.txt")))
        adapter.clear_workspace()

        self.assertFalse(os.path.exists(workspace_dir))
        self.assertFalse(os.path.exists(os.path.join(workspace_dir, "test.txt")))


class TestFilesystemStorageAdapter(unittest.TestCase):
    def setUp(self):
        os.environ["CODEBASE_STORAGE_TYPE"] = "filesystem"
        os.environ["CODEBASE_STORAGE_DIR"] = "data/tests/chroma/storage"
        os.environ["CODEBASE_WORKSPACE_DIR"] = "data/tests/chroma/workspaces"

    def tearDown(self) -> None:
        FilesystemStorageAdapter.clear_all_storage()
        return super().tearDown()

    def test_storage_location(self):
        adapter = FilesystemStorageAdapter(1, "test")
        storage_dir = adapter.get_storage_dir()
        self.assertEqual(storage_dir, os.path.abspath("data/tests/chroma/storage"))

        storage_location = adapter.get_storage_location(1, "test")
        self.assertEqual(storage_location, os.path.abspath("data/tests/chroma/storage/1/test"))

        adapter.clear_all_storage()
        self.assertFalse(os.path.exists(storage_location))

    def test_copy_to_workspace(self):
        adapter = FilesystemStorageAdapter(1, "test")
        storage_location = adapter.get_storage_location(1, "test")

        os.makedirs(storage_location, exist_ok=True)
        with open(os.path.join(storage_location, "test.txt"), "w") as f:
            f.write("test")

        self.assertTrue(adapter.copy_to_workspace())
        self.assertTrue(os.path.exists(adapter.tmpdir))
        self.assertTrue(os.path.exists(os.path.join(adapter.tmpdir, "test.txt")))

    def test_save_to_storage(self):
        adapter = FilesystemStorageAdapter(1, "test")
        storage_location = adapter.get_storage_location(1, "test")

        os.makedirs(adapter.tmpdir, exist_ok=True)
        with open(os.path.join(adapter.tmpdir, "test.txt"), "w") as f:
            f.write("test")

        self.assertTrue(adapter.save_to_storage())
        self.assertTrue(os.path.exists(storage_location))
        self.assertTrue(os.path.exists(os.path.join(storage_location, "test.txt")))


class TestGcsStorageAdapter(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["CODEBASE_STORAGE_TYPE"] = "gcs"
        os.environ["CODEBASE_GCS_BUCKET"] = "seer-chroma"
        os.environ["CODEBASE_GCS_STORAGE_DIR"] = "chroma-test/data/storage"

    @patch("seer.automation.codebase.storage_adapters.storage.Client")
    def test_storage_prefix(self, mock_gcs_client):
        adapter = GcsStorageAdapter(1, "test")
        storage_prefix = adapter.get_storage_prefix(1, "test")
        self.assertEqual(storage_prefix, "chroma-test/data/storage/1/test")

    @patch("seer.automation.codebase.storage_adapters.storage.Client")
    def test_copy_to_workspace(self, mock_gcs_client):
        mock_bucket = mock_gcs_client.return_value.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.name = "test.txt"
        mock_blob.download_to_filename.return_value = None
        mock_bucket.list_blobs.return_value = [mock_blob]

        adapter = GcsStorageAdapter(1, "test")

        self.assertTrue(adapter.copy_to_workspace())
        self.assertTrue(os.path.exists(adapter.tmpdir))
        mock_blob.download_to_filename.assert_called_once()
        self.assertIsNotNone(mock_blob.custom_time)
        mock_blob.patch.assert_called_once()

    @patch("seer.automation.codebase.storage_adapters.storage.Client")
    def test_save_to_storage(self, mock_gcs_client):
        mock_bucket = mock_gcs_client.return_value.bucket.return_value
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_blob.upload_from_filename.return_value = None

        adapter = GcsStorageAdapter(1, "test")
        storage_prefix = adapter.get_storage_prefix(1, "test")

        # Simulate files in the workspace
        os.makedirs(adapter.tmpdir, exist_ok=True)
        test_file_path = os.path.join(adapter.tmpdir, "test_file.txt")
        with open(test_file_path, "w") as f:
            f.write("This is a test file.")

        self.assertTrue(adapter.save_to_storage())

        # Verify that the blob upload method was called with the correct path
        mock_bucket.blob.assert_called_with(f"{storage_prefix}/test_file.txt")
        self.assertIsNotNone(mock_blob.custom_time)
        mock_blob.upload_from_filename.assert_called_with(test_file_path)
