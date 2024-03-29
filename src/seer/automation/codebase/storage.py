import dataclasses
import itertools
import json
import logging
import os
import shutil
from typing import Self, Sequence

import chromadb
import numpy as np
import sqlalchemy.orm
from chromadb.api import ClientAPI
from sqlalchemy import and_, delete, select, update
from sqlalchemy.exc import IntegrityError

from seer.automation.autofix.models import RepoDefinition
from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.models import (
    BaseDocumentChunk,
    ChunkQueryResult,
    CodebaseNamespace,
    EmbeddedDocumentChunk,
    RepositoryInfo,
)
from seer.automation.codebase.utils import cleanup_dir
from seer.db import (
    DbCodebaseNamespace,
    DbDocumentChunk,
    DbDocumentTombstone,
    DbRepositoryInfo,
    Session,
)


@dataclasses.dataclass
class CodebaseIndexStorage:
    repo_info: RepositoryInfo
    namespace: CodebaseNamespace
    client: ClientAPI

    @staticmethod
    def get_workspace_location(repo_id: int, namespace_id: int):
        return f"../data/chroma/workspaces/{repo_id}/{namespace_id}"

    @staticmethod
    def get_storage_location(repo_id: int, namespace_slug: str):
        return f"../data/chroma/storage/{repo_id}/{namespace_slug}"

    @staticmethod
    def copy_to_workspace(repo_id: int, namespace_id: int, namespace_slug: str):
        workspace_path = CodebaseIndexStorage.get_workspace_location(repo_id, namespace_id)
        storage_path = CodebaseIndexStorage.get_storage_location(repo_id, namespace_slug)

        if os.path.exists(storage_path):
            shutil.copytree(storage_path, workspace_path, dirs_exist_ok=True)
            return True

        return False

    def __init__(self, repo_info: RepositoryInfo, namespace: CodebaseNamespace):
        self.repo_info = repo_info
        self.namespace = namespace
        self.client = chromadb.PersistentClient(
            path=self.get_workspace_location(namespace.repo_id, namespace.id)
        )

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
        
        did_copy = cls.copy_to_workspace(namespace.repo_id, namespace.id, namespace.slug)
        if did_copy:
            return cls(repo_info, namespace)
        return None

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

        did_copy = cls.copy_to_workspace(namespace.repo_id, namespace.id, namespace.slug)
        if did_copy:
            return cls(repo_info, namespace)
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

        return cls(repo_info, namespace)

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

        return cls(repo_info, namespace)

    def insert_chunks(self, chunks: list[EmbeddedDocumentChunk]):
        collection = self.client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine"}
        )

        BATCH_SIZE = 32768

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(chunk.hash)
            embeddings.append(chunk.embedding.tolist())
            metadatas.append(chunk.get_db_metadata())

            if len(ids) == BATCH_SIZE:
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                ids = []
                embeddings = []
                metadatas = []

        if ids:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

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
        hashes = query_results["ids"][0]
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
        workspace_path = self.get_workspace_location(self.namespace.repo_id, self.namespace.id)
        storage_path = self.get_storage_location(self.namespace.repo_id, self.namespace.slug)
        shutil.copytree(workspace_path, storage_path, dirs_exist_ok=True)

        with Session() as session:
            db_repo_info = self.repo_info.to_db_model()
            db_namespace = self.namespace.to_db_model()

            session.merge(db_repo_info)
            session.merge(db_namespace)
            session.commit()

        autofix_logger.info(f"Saved workspace for namespace {self.namespace.id}")

    def cleanup(self):
        workspace_path = self.get_workspace_location(self.namespace.repo_id, self.namespace.id)

        if os.path.exists(workspace_path):
            cleanup_dir(workspace_path)

        autofix_logger.info(f"Cleaned up workspace for namespace {self.namespace.id}")

    def __del__(self):
        self.cleanup()

    def _local_paths_of(self, chunks: Sequence[BaseDocumentChunk]) -> list[str]:
        return list(set(chunk.path for chunk in chunks))

    def _remove_chunks(self, chunks: Sequence[BaseDocumentChunk], session: sqlalchemy.orm.Session):
        session.query(DbDocumentChunk).filter(
            DbDocumentChunk.repo_id == self.repo_id,
            DbDocumentChunk.namespace == self.namespace,
            DbDocumentChunk.path.in_(self._local_paths_of(chunks)),
        ).delete(synchronize_session=False)

    def replace_documents(
        self,
        chunks: list[EmbeddedDocumentChunk],
        session: sqlalchemy.orm.Session,
    ):
        """
        Removes all tombstones and chunks associated with the given chunks' document objects in this index storage's
        namespace, and then adds them to the db.  Does not synchronize those changes back to the session -- to see the
        changes locally, force new queries.
        """
        self._remove_chunks(chunks, session)

        session.query(DbDocumentTombstone).filter(
            and_(
                DbDocumentTombstone.repo_id == self.repo_id,
                DbDocumentTombstone.namespace == self.namespace,
                DbDocumentTombstone.path.in_(self._local_paths_of(chunks)),
            )
        ).delete(synchronize_session=False)

        session.add_all(
            DbDocumentChunk(
                repo_id=self.repo_id,
                path=chunk.path,
                language=chunk.language,
                index=chunk.index,
                hash=chunk.hash,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
                namespace=self.namespace,
            )
            for chunk in chunks
        )

    def find_documents(
        self, paths: list[str], session: sqlalchemy.orm.Session
    ) -> dict[str, list[DbDocumentChunk]]:
        deleted_paths = session.execute(
            select(DbDocumentTombstone.path).filter(
                DbDocumentTombstone.repo_id == self.repo_id,
                DbDocumentTombstone.namespace == self.namespace,
                DbDocumentTombstone.path.in_(paths),
            )
        ).scalars()
        undeleted_paths = set(paths) - set(deleted_paths)

        local_chunks: list[DbDocumentChunk] = list(
            session.execute(
                select(DbDocumentChunk)
                .filter(
                    DbDocumentChunk.repo_id == self.repo_id,
                    DbDocumentChunk.namespace == self.namespace,
                    DbDocumentChunk.path.in_(undeleted_paths),
                )
                .order_by(DbDocumentChunk.path, DbDocumentChunk.index)
            ).scalars()
        )

        local_paths = set(c.path for c in local_chunks)

        canonical_paths = undeleted_paths - local_paths

        canonical_chunks: list[DbDocumentChunk] = list(
            session.execute(
                select(DbDocumentChunk)
                .filter(
                    DbDocumentChunk.repo_id == self.repo_id,
                    DbDocumentChunk.namespace.is_(None),
                    DbDocumentChunk.path.in_(canonical_paths),
                )
                .order_by(DbDocumentChunk.path, DbDocumentChunk.index)
            ).scalars()
        )

        result = {}
        for path, chunks in itertools.groupby(canonical_chunks, lambda c: c.path):
            result[path] = list(chunks)
        for path, chunks in itertools.groupby(local_chunks, lambda c: c.path):
            result[path] = list(chunks)

        return result

    def apply_namespace(self, sha: str, session: sqlalchemy.orm.Session):
        session.execute(
            select(DbRepositoryInfo).filter(DbRepositoryInfo.id == self.repo_id).with_for_update()
        )

        session.execute(
            delete(DbDocumentChunk).filter(
                DbDocumentChunk.repo_id == self.repo_id,
                DbDocumentChunk.namespace.is_(None),
                DbDocumentChunk.path.in_(
                    select(DbDocumentChunk.path)
                    .filter(
                        DbDocumentChunk.repo_id == self.repo_id,
                        DbDocumentChunk.namespace == self.namespace,
                    )
                    .distinct()
                    .union(
                        select(DbDocumentTombstone.path).filter(
                            DbDocumentTombstone.repo_id == self.repo_id,
                            DbDocumentTombstone.namespace == self.namespace,
                        )
                    )
                ),
            )
        )

        session.execute(
            update(DbDocumentChunk)
            .filter(
                DbDocumentChunk.repo_id == self.repo_id, DbDocumentChunk.namespace == self.namespace
            )
            .values({DbDocumentChunk.namespace: None})
        )

        session.execute(
            delete(DbDocumentTombstone).filter(
                DbDocumentTombstone.repo_id == self.repo_id,
                DbDocumentTombstone.namespace == self.namespace,
            )
        )

        session.execute(
            update(DbRepositoryInfo)
            .filter(DbRepositoryInfo.id == self.repo_id)
            .values({DbRepositoryInfo.sha: sha})
        )

    @classmethod
    def ensure_codebase(
        cls,
        organization: int,
        project: int,
        repo_definition: RepoDefinition,
        namespace: str,
    ) -> Self:
        with Session() as session:
            db_info = DbRepositoryInfo(
                organization=organization,
                project=project,
                provider=repo_definition.provider,
                external_slug=repo_definition.full_name,
                sha="",
            )
            session.add(db_info)

            try:
                session.commit()
            except IntegrityError as e:
                session.rollback()

                maybe_db_info = (
                    session.query(DbRepositoryInfo)
                    .where(
                        DbRepositoryInfo.organization == organization,
                        DbRepositoryInfo.project == project,
                        DbRepositoryInfo.provider == repo_definition.provider,
                        DbRepositoryInfo.external_slug == repo_definition.full_name,
                    )
                    .one_or_none()
                )
                if maybe_db_info is None:
                    # If we can't recover the existing repository info, the original
                    # integrity error has the most context on why, so raise that.
                    raise e
                db_info = maybe_db_info

            return cls(repo_id=db_info.id, namespace=namespace)
