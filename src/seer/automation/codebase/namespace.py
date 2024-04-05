from typing import Literal, Self

import chromadb
import numpy as np
from chromadb.api import API as ChromaClient

from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.models import (
    BaseDocumentChunk,
    ChunkQueryResult,
    CodebaseNamespace,
    EmbeddedDocumentChunk,
    RepositoryInfo,
)
from seer.automation.codebase.storage_adapters import StorageAdapter, get_storage_adapter_class
from seer.automation.models import RepoDefinition
from seer.db import DbCodebaseNamespace, DbRepositoryInfo, Session


class CodebaseNamespaceManager:
    repo_info: RepositoryInfo
    namespace: CodebaseNamespace
    client: ChromaClient
    storage_adapter: StorageAdapter

    def __init__(
        self,
        repo_info: RepositoryInfo,
        namespace: CodebaseNamespace,
        storage_adapter: StorageAdapter,
    ):
        self.repo_info = repo_info
        self.namespace = namespace
        self.storage_adapter = storage_adapter
        self.client = chromadb.PersistentClient(
            path=storage_adapter.get_workspace_location(namespace.repo_id, namespace.id)
        )

    @classmethod
    def load_workspace(cls, namespace_id: int):
        with Session() as session:
            db_repo_info = session.get(DbRepositoryInfo, namespace_id)

            if db_repo_info is None:
                raise ValueError(f"Repository with id {namespace_id} not found")

            db_namespace = session.get(DbCodebaseNamespace, namespace_id)

            if db_namespace is None:
                raise ValueError(f"Repository namespace with id {namespace_id} not found")

            repo_info = RepositoryInfo.from_db(db_repo_info)
            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = get_storage_adapter_class()(repo_info.id, namespace.id, namespace.slug)
        did_copy = storage_adapter.copy_to_workspace()
        if did_copy:
            return cls(repo_info, namespace, storage_adapter)
        return None

    @classmethod
    def load_workspace_for_repo_definition(
        cls,
        organization: int,
        project: int,
        repo: RepoDefinition,
        sha: str | None = None,
        tracking_branch: str | None = None,
    ) -> Self | None:
        with Session() as session:
            db_repo_info = (
                session.query(DbRepositoryInfo)
                .filter(
                    DbRepositoryInfo.organization == organization,
                    DbRepositoryInfo.project == project,
                    DbRepositoryInfo.provider == repo.provider,
                    DbRepositoryInfo.external_slug == repo.full_name,
                )
                .one_or_none()
            )

            if db_repo_info is None:
                return None

            db_namespace = None
            if sha:
                db_namespace = (
                    session.query(DbCodebaseNamespace)
                    .filter(
                        DbCodebaseNamespace.repo_id == db_repo_info.id,
                        DbCodebaseNamespace.sha == sha,
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
                db_namespace = session.get(DbCodebaseNamespace, db_repo_info.default_namespace)

            if db_namespace is None:
                return None

            repo_info = RepositoryInfo.from_db(db_repo_info)
            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = get_storage_adapter_class()(repo_info.id, namespace.id, namespace.slug)
        did_copy = storage_adapter.copy_to_workspace()
        if did_copy:
            return cls(repo_info, namespace, storage_adapter)
        return None

    @classmethod
    def create_repo(
        cls,
        organization: int,
        project: int,
        provider: str,
        external_slug: str,
        head_sha: str,
        tracking_branch: str | None = None,
    ):
        with Session() as session:
            db_repo_info = DbRepositoryInfo(
                organization=organization,
                project=project,
                provider=provider,
                external_slug=external_slug,
            )
            session.add(db_repo_info)
            session.flush()

            db_namespace = DbCodebaseNamespace(
                repo_id=db_repo_info.id,
                sha=head_sha,
                tracking_branch=tracking_branch,
            )

            session.add(db_namespace)
            session.flush()

            db_repo_info.default_namespace = db_namespace.id

            session.commit()

            repo_info = RepositoryInfo.from_db(db_repo_info)
            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = get_storage_adapter_class()(repo_info.id, namespace.id, namespace.slug)

        return cls(repo_info, namespace, storage_adapter)

    @classmethod
    def get_or_create_namespace_for_repo(
        cls,
        organization: int,
        project: int,
        provider: str,
        external_slug: str,
        head_sha: str,
        tracking_branch: str | None = None,
    ):
        with Session() as session:
            db_repo_info = (
                session.query(DbRepositoryInfo)
                .filter(
                    DbRepositoryInfo.organization == organization,
                    DbRepositoryInfo.project == project,
                    DbRepositoryInfo.provider == provider,
                    DbRepositoryInfo.external_slug == external_slug,
                )
                .one_or_none()
            )

            if db_repo_info is None:
                return cls.create_repo(
                    organization=organization,
                    project=project,
                    provider=provider,
                    external_slug=external_slug,
                    head_sha=head_sha,
                    tracking_branch=tracking_branch,
                )

            return cls.create_namespace_for_repo(
                repo_id=db_repo_info.id,
                sha=head_sha,
                tracking_branch=tracking_branch,
            )

    @classmethod
    def create_namespace_for_repo(
        cls,
        repo_id: int,
        sha: str,
        tracking_branch: str | None = None,
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

            if existing_namespace:
                raise ValueError(
                    f"Namespace for repository {repo_id} with sha {sha} and tracking branch {tracking_branch} already exists"
                )

            db_repo_info = session.get(DbRepositoryInfo, repo_id)

            if db_repo_info is None:
                raise ValueError(f"Repository with id {repo_id} not found")

            db_namespace = DbCodebaseNamespace(
                repo_id=repo_id,
                sha=sha,
                tracking_branch=tracking_branch,
            )

            session.add(db_namespace)
            session.commit()

            repo_info = RepositoryInfo.from_db(db_repo_info)
            namespace = CodebaseNamespace.from_db(db_namespace)

        storage_adapter = get_storage_adapter_class()(repo_info.id, namespace.id, namespace.slug)

        return cls(repo_info, namespace, storage_adapter)

    def chunk_hashes_exist(self, hashes: list[str]):
        collection = self.client.get_collection("chunks")

        results = collection.get(
            ids=hashes,
        )

        ids = results["ids"]

        return ids

    def insert_chunks(self, chunks: list[EmbeddedDocumentChunk]):
        collection = self.client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine"}
        )

        # Chroma has a maximum 34k add limit per time, batched to 32768 because I like 2^x numbers
        BATCH_SIZE = 32768

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

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
        collection = self.client.get_collection("chunks")

        collection.delete(where={"path": {"$in": paths}})  # type: ignore

    def delete_chunks(self, chunk_hashes: list[str]):
        collection = self.client.get_collection("chunks")

        collection.delete(ids=chunk_hashes)

    def update_chunks_metadata(self, chunks: list[BaseDocumentChunk]):
        collection = self.client.get_collection("chunks")

        metadatas = []
        for chunk in chunks:
            metadatas.append(chunk.get_db_metadata())

        collection.update(
            ids=[chunk.hash for chunk in chunks],
            metadatas=metadatas,
        )

    def get_chunks_for_paths(self, paths: list[str]) -> list[ChunkQueryResult]:
        collection = self.client.get_collection("chunks")

        results = collection.query(
            where={"path": {"$in": paths}},  # type: ignore
        )

        return self._get_chunk_query_results(results)

    def _get_chunk_query_results(
        self, query_results: chromadb.QueryResult
    ) -> list[ChunkQueryResult]:
        if not query_results["ids"]:
            return []

        hashes = query_results["ids"][0]  # Assumes a single query
        metadatas = query_results["metadatas"]

        if not metadatas:
            return []

        paths = [str(metadata["path"]) for metadata in metadatas[0]]
        languages = [str(metadata["language"]) for metadata in metadatas[0]]

        return [
            ChunkQueryResult(path=path, hash=hash, language=language)
            for path, hash, language in zip(paths, hashes, languages)
        ]

    def query_chunks(self, query_embedding: np.ndarray, top_k: int = 10) -> list[ChunkQueryResult]:
        collection = self.client.get_collection("chunks")

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        return self._get_chunk_query_results(results)

    def save(self):
        if not self.storage_adapter.save_to_storage():
            autofix_logger.error(f"Failed to save workspace for namespace {self.namespace.id}")
            return

        with Session() as session:
            db_repo_info = self.repo_info.to_db_model()
            db_namespace = self.namespace.to_db_model()

            session.merge(db_repo_info)
            session.merge(db_namespace)
            session.commit()

        autofix_logger.info(f"Saved workspace for namespace {self.namespace.id}")

    def cleanup(self):
        self.storage_adapter.cleanup()

        autofix_logger.info(f"Cleaned up workspace for namespace {self.namespace.id}")

    def __del__(self):
        self.cleanup()
