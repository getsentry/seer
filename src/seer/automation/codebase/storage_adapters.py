import abc
import dataclasses
import os
import shutil

from google.cloud import storage

from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.utils import cleanup_dir


@dataclasses.dataclass
class StorageAdapter(abc.ABC):
    repo_id: int
    namespace_id: int
    namespace_slug: str

    @staticmethod
    def get_workspace_location(repo_id: int, namespace_id: int):
        workspace_dir = os.getenv("CODEBASE_WORKSPACE_DIR", "data/chroma/workspaces")
        return os.path.abspath(os.path.join(workspace_dir, f"{repo_id}/{namespace_id}"))

    @abc.abstractmethod
    def copy_to_workspace(self) -> bool:
        pass

    @abc.abstractmethod
    def save_to_storage(self) -> bool:
        pass

    def cleanup(self):
        workspace_path = self.get_workspace_location(self.repo_id, self.namespace_id)

        if os.path.exists(workspace_path):
            cleanup_dir(workspace_path)


class FilesystemStorageAdapter(StorageAdapter):
    """
    A storage adapter designed to store database files on the filesystem.
    This adapter should be used when you want to store your database files
    locally within the filesystem, providing a straightforward way to manage
    data storage and retrieval from a local directory structure.
    """

    @staticmethod
    def get_storage_location(repo_id: int, namespace_slug: str):
        storage_dir = os.getenv("CODEBASE_STORAGE_DIR", "data/chroma/storage")
        return os.path.abspath(os.path.join(storage_dir, f"{repo_id}/{namespace_slug}"))

    def copy_to_workspace(self):
        workspace_path = self.get_workspace_location(self.repo_id, self.namespace_id)
        storage_path = self.get_storage_location(self.repo_id, self.namespace_slug)

        shutil.copytree(storage_path, workspace_path, dirs_exist_ok=True)

        return True

    def save_to_storage(self):
        workspace_path = self.get_workspace_location(self.repo_id, self.namespace_id)
        storage_path = self.get_storage_location(self.repo_id, self.namespace_slug)
        shutil.copytree(workspace_path, storage_path, dirs_exist_ok=True)

        return True


class GcsStorageAdapter(StorageAdapter):
    """
    A storage adapter designed to store database files in Google Cloud Storage.
    """

    storage_client = storage.Client(project="super-big-data")
    bucket = storage_client.bucket("sentry-ml")

    @staticmethod
    def get_storage_prefix(repo_id: int, namespace_slug: str):
        storage_dir = os.getenv("CODEBASE_GCS_STORAGE_DIR", "tmp_jenn/dev/chroma/storage")
        return os.path.join(storage_dir, f"{repo_id}/{namespace_slug}")

    def copy_to_workspace(self) -> bool:
        workspace_path = self.get_workspace_location(self.repo_id, self.namespace_id)
        storage_prefix = self.get_storage_prefix(self.repo_id, self.namespace_slug)

        blobs = self.bucket.list_blobs(prefix=storage_prefix)
        blobs_list = list(blobs)
        print("LISTED BLOBS:", blobs_list)
        for blob in blobs_list:
            filename = blob.name.replace(storage_prefix + "/", "")
            download_path = os.path.join(workspace_path, filename)

            if not os.path.exists(os.path.dirname(download_path)):
                os.makedirs(os.path.dirname(download_path))

            blob.download_to_filename(download_path)
            print("downloaded file:", filename, "to", download_path)

        autofix_logger.debug(
            f"Downloaded files from {storage_prefix} to workspace: {workspace_path}"
        )

        return True

    def save_to_storage(self) -> bool:
        workspace_path = self.get_workspace_location(self.repo_id, self.namespace_id)
        storage_prefix = self.get_storage_prefix(self.repo_id, self.namespace_slug)

        blobs = self.bucket.list_blobs(prefix=storage_prefix)
        for blob in blobs:
            blob.delete()

        for root, dirs, files in os.walk(workspace_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, workspace_path)
                blob_path = f"{storage_prefix}/{relative_path}"
                blob = self.bucket.blob(blob_path)
                blob.upload_from_filename(file_path)

        autofix_logger.debug(f"Uploaded files from workspace: {workspace_path} to {storage_prefix}")

        return True


def get_storage_adapter_class() -> type[StorageAdapter]:
    storage_type = os.getenv("CODEBASE_STORAGE_TYPE", "filesystem")

    autofix_logger.debug(f"Using storage type: {storage_type}")

    if storage_type == "filesystem":
        return FilesystemStorageAdapter
    elif storage_type == "gcs":
        return GcsStorageAdapter
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
