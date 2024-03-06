import dataclasses
import logging
from collections import defaultdict
from typing import Self

import sqlalchemy.orm
from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from seer.automation.autofix.models import RepoDefinition
from seer.automation.codebase.models import DocumentChunk, DocumentChunkWithEmbedding
from seer.db import DbDocumentChunk, DbDocumentTombstone, DbRepositoryInfo

logger = logging.getLogger("autofix")


@dataclasses.dataclass
class CodebaseIndexStorage:
    repo_id: int
    namespace: str

    def delete_paths(self, paths: list[str], session: sqlalchemy.orm.Session):
        session.query(DbDocumentChunk).filter(
            DbDocumentChunk.repo_id == self.repo_id,
            DbDocumentChunk.path.in_(paths),
            DbDocumentChunk.namespace == self.namespace,
        ).delete(synchronize_session=False)

        insert_stmt = (
            insert(DbDocumentTombstone)
            .values(
                [
                    {
                        DbDocumentTombstone.repo_id: self.repo_id,
                        DbDocumentTombstone.path: path,
                        DbDocumentTombstone.namespace: self.namespace,
                    }
                    for path in paths
                ]
            )
            .on_conflict_do_nothing()
        )
        session.execute(insert_stmt)

    def _repo_files(self, chunks: list[DocumentChunk]) -> dict[int, list[str]]:
        result = defaultdict(list)
        for chunk in chunks:
            result[chunk.repo_id].append(chunk.path)
        return result

    def replace_documents(
        self, chunks: list[DocumentChunkWithEmbedding], session: sqlalchemy.orm.Session
    ):
        session.query(DbDocumentChunk).filter(
            or_(
                *(
                    and_(
                        DbDocumentChunk.repo_id == repo_id,
                        DbDocumentChunk.namespace == self.namespace,
                        DbDocumentChunk.path.in_(paths),
                    )
                    for repo_id, paths in self._repo_files(chunks).items()
                )
            )
        ).delete(synchronize_session=False)

        session.query(DbDocumentTombstone).filter(
            or_(
                *(
                    and_(
                        DbDocumentTombstone.repo_id == repo_id,
                        DbDocumentTombstone.namespace == self.namespace,
                        DbDocumentTombstone.path.in_(paths),
                    )
                    for repo_id, paths in self._repo_files(chunks).items()
                )
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
        session: sqlalchemy.orm.Session,
    ) -> Self:
        db_info = DbRepositoryInfo(
            organization=organization,
            project=project,
            provider=repo_definition.provider,
            external_slug=repo_definition.external_slug,
            sha="",
        )
        session.add(db_info)

        try:
            session.commit()
        except IntegrityError as e:
            db_info = (
                session.query(DbRepositoryInfo)
                .filter(
                    organization=organization,
                    project=project,
                    provider=repo_definition.provider,
                    external_slug=repo_definition.external_slug,
                )
                .one_or_none()
            )
            if db_info is None:
                # If we can't recover the existing repository info, the original
                # integrity error has the most context on why, so raise that.
                raise e

        return cls(repo_id=db_info.id, namespace=namespace)
