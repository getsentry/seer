import datetime
import logging
import time
from typing import Mapping, Optional, Self

import chromadb
import numpy as np
import sentry_sdk
from chromadb.api import API as ChromaClient

from seer.automation.codebase.models import (
    BaseDocumentChunk,
    ChunkQueryResult,
    ChunkResult,
    CodebaseNamespace,
    CodebaseNamespaceStatus,
    EmbeddedDocumentChunk,
    RepositoryInfo,
)
from seer.automation.codebase.storage_adapters import StorageAdapter
from seer.automation.models import RepoDefinition
from seer.db import DbCodebaseNamespace, DbCodebaseNamespaceMutex, DbRepositoryInfo, Session
from seer.dependency_injection import inject, injected

logger = logging.getLogger(__name__)


class CodebaseNamespaceManager:
    """
    Manages the namespace operations within a codebase, interfacing with both the storage layer and the Chroma database.

    Attributes:
        repo_info (RepositoryInfo): Information about the repository associated with this namespace.
        namespace (CodebaseNamespace): The specific namespace within the codebase being managed.
        client (ChromaClient): Client for interacting with the Chroma database.
        storage_adapter (StorageAdapter): Adapter handling the storage operations for the namespace.
    """

    repo_info: RepositoryInfo
    namespace: CodebaseNamespace
    client: ChromaClient
    storage_adapter: StorageAdapter

    NAMESPACE_MUTEX_TIMEOUT_MINUTES = 2

    def __init__(
        self,
        repo_info: RepositoryInfo,
        namespace: CodebaseNamespace,
        storage_adapter: StorageAdapter,
    ):
        self.repo_info = repo_info
        self.namespace = namespace
        self.storage_adapter = storage_adapter
        self.client = chromadb.PersistentClient(path=storage_adapter.tmpdir)

        self._log_accessed_at()

    def __del__(self):
        self.cleanup()

    @staticmethod
    def get_mutex(namespace_id: int) -> datetime.datetime | None:
        with Session() as session:
            namespace_mutex = (
                session.query(DbCodebaseNamespaceMutex)
                .filter_by(namespace_id=namespace_id)
                .one_or_none()
            )

            return namespace_mutex.created_at if namespace_mutex else None

    @staticmethod
    def _clear_mutex(namespace_id: int):
        logger.debug(f"Mutex for namespace {namespace_id} is being cleared...")
        with Session() as session:
            session.query(DbCodebaseNamespaceMutex).filter_by(namespace_id=namespace_id).delete()
            session.commit()

    @staticmethod
    def _set_mutex(namespace_id: int):
        logger.debug(f"Mutex for namespace {namespace_id} is being set...")
        with Session() as session:
            mutex = DbCodebaseNamespaceMutex(
                namespace_id=namespace_id, created_at=datetime.datetime.now()
            )
            session.add(mutex)
            session.commit()

    @staticmethod
    def _wait_for_mutex_clear(namespace_id: int):
        with sentry_sdk.start_span(
            op="seer.automation.codebase._wait_for_mutex_clear",
            description="Awaiting the mutex associated with a namespace",
        ):
            while namespace_mutex := CodebaseNamespaceManager.get_mutex(namespace_id):
                if (
                    namespace_mutex is not None
                    and datetime.datetime.now() - namespace_mutex
                    > datetime.timedelta(
                        minutes=CodebaseNamespaceManager.NAMESPACE_MUTEX_TIMEOUT_MINUTES
                    )
                ):
                    logger.warning(
                        f"Mutex for namespace {namespace_id} has been held for more than {CodebaseNamespaceManager.NAMESPACE_MUTEX_TIMEOUT_MINUTES} minutes"
                    )
                    CodebaseNamespaceManager._clear_mutex(namespace_id)
                    break

                logger.debug(f"Mutex for namespace {namespace_id} is held, waiting...")
                time.sleep(1)

    @staticmethod
    def get_namespace(
        organization: int,
        project: int,
        repo: RepoDefinition,
        sha: str | None,
        tracking_branch: str | None,
    ):
        with Session() as session:
            db_repo_info = (
                session.query(DbRepositoryInfo)
                .filter(
                    DbRepositoryInfo.organization == organization,
                    DbRepositoryInfo.project == project,
                    DbRepositoryInfo.provider == repo.provider,
                    DbRepositoryInfo.external_id == repo.external_id,
                )
                .one_or_none()
            )

            if db_repo_info is None:
                return None

            db_namespace = None
            if sha is None and tracking_branch is None:
                db_namespace = session.get(DbCodebaseNamespace, db_repo_info.default_namespace)
            elif sha is not None:
                db_namespace = (
                    session.query(DbCodebaseNamespace)
                    .filter(
                        DbCodebaseNamespace.repo_id == db_repo_info.id,
                        DbCodebaseNamespace.sha == sha,
                    )
                    .one_or_none()
                )

            elif tracking_branch is not None:
                db_namespace = (
                    session.query(DbCodebaseNamespace)
                    .filter(
                        DbCodebaseNamespace.repo_id == db_repo_info.id,
                        DbCodebaseNamespace.tracking_branch == tracking_branch,
                    )
                    .one_or_none()
                )

            if db_namespace is None:
                return None

            return CodebaseNamespace.from_db(db_namespace)

    @classmethod
    @inject
    def load_workspace(
        cls,
        namespace_id: int,
        skip_copy: bool = False,
        storage_type: type[StorageAdapter] = injected,
    ):
        with Session() as session:
            db_namespace = session.get(DbCodebaseNamespace, namespace_id)

            if db_namespace is None:
                raise ValueError(f"Repository namespace with id {namespace_id} not found")

            db_repo_info = session.get(DbRepositoryInfo, db_namespace.repo_id)

            if db_repo_info is None:
                raise ValueError(f"Repository with id {db_namespace.repo_id} not found")

            repo_info = RepositoryInfo.from_db(db_repo_info)
            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = storage_type(repo_id=repo_info.id, namespace_slug=namespace.slug)

        did_copy = False
        if not skip_copy:
            cls._wait_for_mutex_clear(namespace.id)
            cls._set_mutex(namespace.id)

            did_copy = storage_adapter.copy_to_workspace()

            cls._clear_mutex(namespace.id)

        if skip_copy or did_copy:
            return cls(repo_info, namespace, storage_adapter)
        return None

    @classmethod
    @inject
    def load_workspace_for_repo_definition(
        cls,
        organization: int,
        project: int,
        repo: RepoDefinition,
        tracking_branch: str | None = None,
        storage_type: type[StorageAdapter] = injected,
    ) -> Self | None:
        with Session() as session:
            db_repo_info = (
                session.query(DbRepositoryInfo)
                .filter(
                    DbRepositoryInfo.organization == organization,
                    DbRepositoryInfo.project == project,
                    DbRepositoryInfo.provider == repo.provider,
                    DbRepositoryInfo.external_id == repo.external_id,
                )
                .one_or_none()
            )

            if db_repo_info is None:
                logger.debug(
                    f"Failed to get repo info for org {organization}, project {project}, external_id {repo.external_id}"
                )
                return None

            db_namespace = None
            if repo.base_commit_sha:
                db_namespace = (
                    session.query(DbCodebaseNamespace)
                    .filter(
                        DbCodebaseNamespace.repo_id == db_repo_info.id,
                        DbCodebaseNamespace.sha == repo.base_commit_sha,
                    )
                    .one_or_none()
                )
            elif tracking_branch:
                db_namespace = (
                    session.query(DbCodebaseNamespace)
                    .filter(
                        DbCodebaseNamespace.repo_id == db_repo_info.id,
                        DbCodebaseNamespace.tracking_branch == tracking_branch,
                    )
                    .one_or_none()
                )
            else:
                if not db_repo_info.default_namespace:
                    return None

                db_namespace = session.get(DbCodebaseNamespace, db_repo_info.default_namespace)

            if db_namespace is None:
                logger.debug(
                    f"Failed to get namespace info for org {organization}, project {project}, external_id {repo.external_id}"
                )
                return None

            repo_info = RepositoryInfo.from_db(db_repo_info)
            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = storage_type(repo_id=repo_info.id, namespace_slug=namespace.slug)

        cls._wait_for_mutex_clear(namespace.id)
        cls._set_mutex(namespace.id)

        did_copy = storage_adapter.copy_to_workspace()

        cls._clear_mutex(namespace.id)

        if did_copy:
            return cls(repo_info, namespace, storage_adapter)
        return None

    @classmethod
    @inject
    def create_repo(
        cls,
        organization: int,
        project: int,
        repo: RepoDefinition,
        head_sha: str,
        tracking_branch: str | None = None,
        should_set_as_default: bool = False,
        storage_type: type[StorageAdapter] = injected,
    ):
        logger.info(
            f"Creating new repo for {organization}/{project}/{repo.external_id} (repo: {repo.full_name})"
        )
        with Session() as session:
            db_repo_info = DbRepositoryInfo(
                organization=organization,
                project=project,
                provider=repo.provider,
                external_slug=repo.full_name,
                external_id=repo.external_id,
            )
            session.add(db_repo_info)
            session.flush()

            db_namespace = DbCodebaseNamespace(
                repo_id=db_repo_info.id,
                sha=head_sha,
                tracking_branch=tracking_branch,
                status=CodebaseNamespaceStatus.PENDING,
            )

            session.add(db_namespace)
            session.flush()

            if should_set_as_default:
                db_repo_info.default_namespace = db_namespace.id

            session.commit()

            repo_info = RepositoryInfo.from_db(db_repo_info)
            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = storage_type(repo_id=repo_info.id, namespace_slug=namespace.slug)

        return cls(repo_info, namespace, storage_adapter)

    @classmethod
    def create_namespace_with_new_or_existing_repo(
        cls,
        organization: int,
        project: int,
        repo: RepoDefinition,
        head_sha: str,
        tracking_branch: str | None = None,
        should_set_as_default: bool = False,
    ):
        with Session() as session:
            db_repo_info = (
                session.query(DbRepositoryInfo)
                .filter(
                    DbRepositoryInfo.organization == organization,
                    DbRepositoryInfo.project == project,
                    DbRepositoryInfo.provider == repo.provider,
                    DbRepositoryInfo.external_id == repo.external_id,
                )
                .one_or_none()
            )

            if db_repo_info is None:
                return cls.create_repo(
                    organization=organization,
                    project=project,
                    repo=repo,
                    head_sha=head_sha,
                    tracking_branch=tracking_branch,
                    should_set_as_default=should_set_as_default,
                )
            return cls.create_or_get_namespace_for_repo(
                repo_id=db_repo_info.id,
                sha=head_sha,
                tracking_branch=tracking_branch,
                should_set_as_default=should_set_as_default,
            )

    @classmethod
    @inject
    def create_or_get_namespace_for_repo(
        cls,
        repo_id: int,
        sha: str,
        tracking_branch: str | None = None,
        should_set_as_default: bool = False,
        storage_type: type[StorageAdapter] = injected,
    ):
        with Session() as session:
            existing_namespace = None
            if tracking_branch:
                existing_namespace = (
                    session.query(DbCodebaseNamespace)
                    .filter_by(repo_id=repo_id, tracking_branch=tracking_branch)
                    .one_or_none()
                )
            else:
                existing_namespace = (
                    session.query(DbCodebaseNamespace)
                    .filter(
                        DbCodebaseNamespace.repo_id == repo_id,
                        DbCodebaseNamespace.sha == sha,
                    )
                    .one_or_none()
                )

            db_repo_info = session.get(DbRepositoryInfo, repo_id)

            if db_repo_info is None:
                raise ValueError(f"Repository with id {repo_id} not found")

            if existing_namespace:
                logger.info(
                    f"Using existing namespace for {db_repo_info.external_id} namespace id: {existing_namespace.id}"
                )
                db_namespace = existing_namespace
            else:
                logger.info(
                    f"Creating namespace with existing repo for {db_repo_info.organization}/{db_repo_info.project}/{db_repo_info.external_id} (repo: {db_repo_info.external_slug})"
                )
                db_namespace = DbCodebaseNamespace(
                    repo_id=repo_id,
                    sha=sha,
                    tracking_branch=tracking_branch,
                    status=CodebaseNamespaceStatus.PENDING,
                )

                session.add(db_namespace)
                session.commit()

            repo_info = RepositoryInfo.from_db(db_repo_info)

            if should_set_as_default:
                repo_info.default_namespace = db_namespace.id
                session.merge(repo_info.to_db_model())
                session.commit()

            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = storage_type(repo_id=repo_info.id, namespace_slug=namespace.slug)

        return cls(repo_info, namespace, storage_adapter)

    @staticmethod
    def does_repo_exist(
        organization: int, project: int, provider: str, external_id: str, sha: Optional[str] = None
    ):
        with Session() as session:
            repo = (
                session.query(DbRepositoryInfo)
                .filter(
                    DbRepositoryInfo.organization == organization,
                    DbRepositoryInfo.project == project,
                    DbRepositoryInfo.provider == provider,
                    DbRepositoryInfo.external_id == external_id,
                )
                .one_or_none()
            )
            if repo is None:
                return False

            if sha:
                return (
                    session.query(DbCodebaseNamespace)
                    .filter(DbCodebaseNamespace.repo_id == repo.id, DbCodebaseNamespace.sha == sha)
                    .count()
                    > 0
                )

            return True

    def is_ready(self):
        """
        Tries to check if the namespace is ready for use. This is a best-effort check and may not be 100% accurate.
        The idea behind this is to check to make sure the namespace is not corrupt and actually loaded.
        """
        try:
            collection = self.client.get_collection("chunks")
        except ValueError:
            # Happens when the collection chunks does not exist.
            # https://sentry.sentry.io/issues/5303418225/?project=6178942
            return False

        if collection.count() == 0:
            return False

        if self.namespace.status != CodebaseNamespaceStatus.CREATED:
            return False

        return True

    def verify_file_integrity(self, paths: set[str]) -> bool:
        """
        Verifies that all the paths in the given set exist in the namespace.
        Heavy operation, should be used sparingly.
        """

        collection = self.client.get_collection("chunks")

        BATCH_SIZE = 32768
        all_retrieved_paths: set[str] = set()
        total_paths = len(paths)
        num_batches = (
            total_paths + BATCH_SIZE - 1
        ) // BATCH_SIZE  # Calculate the number of batches needed

        for i in range(num_batches):
            results = collection.get(
                limit=BATCH_SIZE,
                offset=i * BATCH_SIZE,
            )

            if not results["ids"] or not results["metadatas"]:
                return False

            batch_paths = {str(metadata["path"]) for metadata in results["metadatas"]}
            all_retrieved_paths.update(batch_paths)

        matches = all_retrieved_paths == paths

        if not matches:
            missing_paths = paths - all_retrieved_paths
            logger.debug(f"Paths mismatch: {missing_paths}")

        return matches

    def chunk_hashes_exist(self, hashes: list[str]):
        if not hashes:
            return []

        collection = self.client.get_collection("chunks")

        results = collection.get(
            ids=hashes,
        )

        ids = results["ids"]

        return ids

    def insert_chunks(self, chunks: list[EmbeddedDocumentChunk]):
        if not chunks:
            return

        collection = self.client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine"}
        )

        # Chroma has a maximum 34k add limit per time, batched to 32768 because I like 2^x numbers
        BATCH_SIZE = 32768

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[Mapping] = []

        for chunk in chunks:
            ids.append(chunk.hash)
            embeddings.append(chunk.embedding.tolist())
            metadatas.append(chunk.get_db_metadata())

            if len(ids) == BATCH_SIZE:
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)  # type: ignore
                ids = []
                embeddings = []
                metadatas = []

        if ids:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)  # type: ignore

    def delete_paths(self, paths: list[str]):
        if not paths:
            return

        collection = self.client.get_collection("chunks")

        collection.delete(where={"path": {"$in": paths}})  # type: ignore

    def delete_chunks(self, chunk_hashes: list[str]):
        if not chunk_hashes:
            return

        collection = self.client.get_collection("chunks")

        collection.delete(ids=chunk_hashes)

    def update_chunks_metadata(self, chunks: list[BaseDocumentChunk]):
        if not chunks:
            return

        collection = self.client.get_collection("chunks")

        metadatas: list[Mapping] = []
        for chunk in chunks:
            metadatas.append(chunk.get_db_metadata())

        collection.update(
            ids=[chunk.hash for chunk in chunks],
            metadatas=metadatas,
        )

    def get_chunks_for_paths(self, paths: list[str]) -> list[ChunkResult]:
        if not paths:
            return []

        collection = self.client.get_collection("chunks")

        results = collection.get(
            where={"path": {"$in": paths}},  # type: ignore
        )

        return self._get_chunk_get_results(results)

    def query_chunks(self, query_embedding: np.ndarray, top_k: int = 10) -> list[ChunkQueryResult]:
        collection = self.client.get_collection("chunks")

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        return self._get_chunk_query_results(results)

    def save_records(self):
        with Session() as session:
            db_repo_info = self.repo_info.to_db_model()
            db_namespace = self.namespace.to_db_model()

            db_namespace.updated_at = datetime.datetime.now()
            db_namespace.accessed_at = datetime.datetime.now()

            session.merge(db_repo_info)
            session.merge(db_namespace)
            session.commit()

    def save(self):
        self._wait_for_mutex_clear(self.namespace.id)
        self._set_mutex(self.namespace.id)

        if not self.storage_adapter.save_to_storage():
            logger.error(f"Failed to save workspace for namespace {self.namespace.id}")
            return

        self.save_records()

        self._clear_mutex(self.namespace.id)

        logger.info(f"Saved workspace for namespace {self.namespace.id}")

    def delete(self):
        self._wait_for_mutex_clear(self.namespace.id)
        self._set_mutex(self.namespace.id)

        if not self.storage_adapter.delete_from_storage():
            logger.error(f"Failed to delete workspace for namespace {self.namespace.id}")

        with Session() as session:
            session.query(DbCodebaseNamespaceMutex).filter_by(
                namespace_id=self.namespace.id
            ).delete()
            session.query(DbCodebaseNamespace).filter_by(id=self.namespace.id).delete()

            # Clear default namespace if it was set
            if self.repo_info.default_namespace == self.namespace.id:
                self.repo_info.default_namespace = None
                session.merge(self.repo_info.to_db_model())

            session.commit()

        logger.info(f"Deleted workspace for namespace {self.namespace.id}")

    def cleanup(self):
        self.storage_adapter.clear_workspace()

        logger.info(f"Cleaned up workspace for namespace {self.namespace.id}")

    def _log_accessed_at(self):
        with Session() as session:
            self.namespace.accessed_at = datetime.datetime.now()
            db_namespace = self.namespace.to_db_model()
            session.merge(db_namespace)
            session.commit()

    def _get_chunk_query_results(
        self, query_results: chromadb.QueryResult
    ) -> list[ChunkQueryResult]:
        if not query_results["ids"] or not query_results["distances"]:
            return []

        hashes = query_results["ids"][0]  # Assumes a single query
        distances = query_results["distances"][0]

        metadatas = query_results["metadatas"]

        if not metadatas:
            return []

        paths = [str(metadata["path"]) for metadata in metadatas[0]]
        indicies = [int(metadata["index"]) for metadata in metadatas[0]]
        languages = [str(metadata["language"]) for metadata in metadatas[0]]

        return [
            ChunkQueryResult(
                path=path, hash=hash, language=language, index=index, distance=distance
            )
            for path, hash, language, index, distance in zip(
                paths, hashes, languages, indicies, distances
            )
        ]

    def _get_chunk_get_results(self, query_results: chromadb.GetResult) -> list[ChunkResult]:
        if not query_results["ids"]:
            return []

        hashes = query_results["ids"]  # Assumes a single query
        metadatas = query_results["metadatas"]

        if not metadatas:
            return []

        paths = [str(metadata["path"]) for metadata in metadatas]
        indicies = [int(metadata["index"]) for metadata in metadatas]
        languages = [str(metadata["language"]) for metadata in metadatas]

        return [
            ChunkResult(path=path, hash=hash, language=language, index=index)
            for path, hash, language, index in zip(paths, hashes, languages, indicies)
        ]
