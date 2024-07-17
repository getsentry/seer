import abc
import datetime
import os
import shutil
import tempfile

from google.cloud import storage
from google.cloud.storage import Bucket

from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.utils import cleanup_dir
from seer.bootup import module
from seer.configuration import AppConfig, CodebaseStorageType
from seer.dependency_injection import inject, injected


class StorageAdapter(abc.ABC):
    repo_id: int
    namespace_slug: str
    app_config: AppConfig

    @inject
    def __init__(self, repo_id: int, namespace_slug: str, app_config: AppConfig = injected):
        self.repo_id = repo_id
        self.namespace_slug = namespace_slug
        self.tmpdir = tempfile.mkdtemp()
        self.app_config = app_config

    def __del__(self):
        self.clear_workspace()

    def clear_workspace(self):
        try:
            cleanup_dir(self.tmpdir)
        except Exception as e:
            autofix_logger.exception(e)

    @abc.abstractmethod
    def copy_to_workspace(self) -> bool:
        pass

    @abc.abstractmethod
    def save_to_storage(self) -> bool:
        pass

    @abc.abstractmethod
    def delete_from_storage(self) -> bool:
        pass


class FilesystemStorageAdapter(StorageAdapter):
    """
    A storage adapter designed to store database files on the filesystem.
    This adapter should be used when you want to store your database files
    locally within the filesystem, providing a straightforward way to manage
    data storage and retrieval from a local directory structure.
    """

    def get_storage_dir(self):
        return self.app_config.CODEBASE_STORAGE_DIR

    def get_storage_location(self):
        storage_dir = self.get_storage_dir()
        return os.path.join(storage_dir, f"{self.repo_id}/{self.namespace_slug}")

    @staticmethod
    @inject
    def clear_all_storage(app_config: AppConfig = injected):
        shutil.rmtree(app_config.CODEBASE_STORAGE_DIR, ignore_errors=True)

    def copy_to_workspace(self):
        storage_path = self.get_storage_location()

        if os.path.exists(storage_path):
            try:
                shutil.copytree(storage_path, self.tmpdir, dirs_exist_ok=True)
            except Exception as e:
                autofix_logger.exception(e)
                return False

        return True

    def save_to_storage(self):
        storage_path = self.get_storage_location()

        if os.path.exists(storage_path):
            try:
                shutil.rmtree(storage_path, ignore_errors=False)
            except Exception as e:
                autofix_logger.exception(e)
                return False

        shutil.copytree(self.tmpdir, storage_path, dirs_exist_ok=True)

        return True

    def delete_from_storage(self):
        storage_path = self.get_storage_location()

        if os.path.exists(storage_path):
            try:
                shutil.rmtree(storage_path, ignore_errors=False)
            except Exception as e:
                autofix_logger.exception(e)
                return False

        return True


class GcsStorageAdapter(StorageAdapter):
    """
    A storage adapter designed to store database files in Google Cloud Storage.
    """

    def get_bucket(self) -> Bucket:
        return storage.Client().bucket(self.app_config.CODEBASE_GCS_STORAGE_BUCKET)

    def get_storage_prefix(self, repo_id: int, namespace_slug: str):
        storage_dir = self.app_config.CODEBASE_GCS_STORAGE_DIR
        return os.path.join(storage_dir, f"{repo_id}/{namespace_slug}")

    def copy_to_workspace(self) -> bool:
        storage_prefix = self.get_storage_prefix(self.repo_id, self.namespace_slug)

        try:
            blobs = self.get_bucket().list_blobs(prefix=storage_prefix)
            blobs_list: list[storage.Blob] = list(blobs)
            for blob in blobs_list:
                if blob.name:
                    filename = blob.name.replace(storage_prefix + "/", "")
                    download_path = os.path.join(self.tmpdir, filename)

                    if not os.path.exists(os.path.dirname(download_path)):
                        os.makedirs(os.path.dirname(download_path))

                    blob.download_to_filename(download_path)

                    # Update the custom time of the blob to the current time
                    # We use custom time to track when the file was last used
                    # This is to manage the lifecycle of the files in the storage
                    blob.custom_time = datetime.datetime.now()
                    blob.patch()

            autofix_logger.debug(
                f"Downloaded files from {storage_prefix} to workspace: {self.tmpdir}"
            )
        except Exception as e:
            autofix_logger.exception(e)
            return False

        return True

    def save_to_storage(self) -> bool:
        storage_prefix = self.get_storage_prefix(self.repo_id, self.namespace_slug)

        try:
            blobs = self.get_bucket().list_blobs(prefix=storage_prefix)
            for blob in blobs:
                # Delete the existing blobs in the storage prefix
                blob.delete()

            for root, dirs, files in os.walk(self.tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.tmpdir)
                    blob_path = f"{storage_prefix}/{relative_path}"

                    blob = self.get_bucket().blob(blob_path)

                    # Update the custom time of the blob to the current time
                    # We use custom time to track when the file was last used
                    # This is to manage the lifecycle of the files in the storage
                    blob.custom_time = datetime.datetime.now()

                    blob.upload_from_filename(file_path)

            autofix_logger.debug(
                f"Uploaded files from workspace: {self.tmpdir} to {storage_prefix}"
            )
        except Exception as e:
            autofix_logger.exception(e)
            return False

        return True

    def delete_from_storage(self) -> bool:
        storage_prefix = self.get_storage_prefix(self.repo_id, self.namespace_slug)

        try:
            blobs = list(self.get_bucket().list_blobs(prefix=storage_prefix))
            self.get_bucket().delete_blobs(blobs)
        except Exception as e:
            autofix_logger.exception(e)
            return False

        return True


@module.provider
def get_storage_adapter_class(config: AppConfig = injected) -> type[StorageAdapter]:
    autofix_logger.debug(f"Using storage type: {config.CODEBASE_STORAGE_TYPE}")

    if config.CODEBASE_STORAGE_TYPE == CodebaseStorageType.FILESYSTEM:
        return FilesystemStorageAdapter
    elif config.CODEBASE_STORAGE_TYPE == CodebaseStorageType.GCS:
        return GcsStorageAdapter
    else:
        raise ValueError(f"Unknown storage type: {config.CODEBASE_STORAGE_TYPE}")
