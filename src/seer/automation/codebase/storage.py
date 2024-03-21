import dataclasses
import itertools
import logging
from collections import defaultdict
from typing import Self, Sequence

import sqlalchemy.orm
from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from seer.automation.autofix.models import RepoDefinition
from seer.automation.codebase.models import BaseDocumentChunk, EmbeddedDocumentChunk
from seer.db import DbDocumentChunk, DbDocumentTombstone, DbRepositoryInfo, Session

logger = logging.getLogger("autofix")


@dataclasses.dataclass
class CodebaseIndexStorage:
    repo_id: int
    namespace: str

    def delete_paths(self, paths: list[str], session: sqlalchemy.orm.Session):
        if not paths:
            return

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
                        "repo_id": self.repo_id,
                        "path": path,
                        "namespace": self.namespace,
                    }
                    for path in paths
                ]
            )
            .on_conflict_do_nothing()
        )
        session.execute(insert_stmt)

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
