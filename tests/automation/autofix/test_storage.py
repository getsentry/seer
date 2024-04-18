import contextlib
import dataclasses
import hashlib
from functools import cached_property
from random import Random
from typing import Annotated

from johen import change_watcher, gen
from johen.examples import Examples
from johen.generators import specialized
from johen.generators.specialized import AsciiWord, FilePath, UnsignedInt
from johen.pytest import parametrize, sometimes
from sqlalchemy import select
from sqlalchemy.sql.functions import count

from seer.automation.autofix.models import RepoDefinition
from seer.automation.codebase.models import BaseDocumentChunk, EmbeddedDocumentChunk
from seer.automation.codebase.storage import CodebaseIndexStorage
from seer.db import DbDocumentChunk, DbDocumentTombstone, DbRepositoryInfo, Session


@dataclasses.dataclass
class EmbeddedDocumentFactory:
    _chunks: tuple[EmbeddedDocumentChunk, ...]

    def as_embedded_chunks(self, file_path: str) -> list[EmbeddedDocumentChunk]:
        assert self._chunks
        return [
            c.model_copy(
                update=dict(
                    id=None,
                    hash=hashlib.sha1(c.content.encode("utf8")).hexdigest(),
                    index=i,
                    path=file_path,
                    token_count=int(len(c.content) / 3),  # approximation
                )
            )
            for i, c in enumerate(self._chunks)
        ]


@dataclasses.dataclass
class CodebaseWithEdits:
    organization_id: Annotated[int, Examples(abs(r.getrandbits(16)) for r in gen)]
    project_id: Annotated[int, Examples(abs(r.getrandbits(16)) for r in gen)]
    repo_definition: RepoDefinition
    namespace: AsciiWord
    documents: dict[FilePath, EmbeddedDocumentFactory]

    @property
    def ordered_paths(self) -> list[FilePath]:
        return sorted(self.documents.keys())

    @property
    def ordered_chunks(self) -> dict[FilePath, list[EmbeddedDocumentChunk]]:
        return {k: v.as_embedded_chunks(k) for k, v in self.documents.items()}

    @property
    def all_chunks(self) -> list[EmbeddedDocumentChunk]:
        return sorted(
            [item for _, chunks in self.ordered_chunks.items() for item in chunks],
            key=lambda item: (item.path, item.index),
        )

    def ensure(self):
        return CodebaseIndexStorage.ensure_codebase(
            organization=self.organization_id,
            project=self.project_id,
            repo_definition=self.repo_definition,
            namespace=self.namespace,
        )

    def apply(self):
        storage = self.ensure()
        with Session() as session:
            storage.replace_documents(
                self.all_chunks,
                session,
            )

            session.commit()
        return storage

    def current_db_chunks(self) -> list[DbDocumentChunk]:
        storage = self.ensure()
        with Session() as session:
            return list(
                session.execute(
                    select(DbDocumentChunk)
                    .filter(
                        DbDocumentChunk.repo_id == storage.repo_id,
                        DbDocumentChunk.namespace == storage.namespace,
                    )
                    .order_by(DbDocumentChunk.path, DbDocumentChunk.index)
                ).scalars()
            )

    def tombstone_paths(self) -> list[str]:
        storage = self.ensure()
        with Session() as session:
            return list(
                session.execute(
                    select(DbDocumentTombstone.path)
                    .filter(
                        DbDocumentTombstone.repo_id == storage.repo_id,
                        DbDocumentTombstone.namespace == storage.namespace,
                    )
                    .order_by(DbDocumentTombstone.path)
                ).scalars()
            )


@dataclasses.dataclass
class FollowupUpdate:
    selector: UnsignedInt | FilePath
    replacement: EmbeddedDocumentFactory | None

    def select(self, codebase: CodebaseWithEdits) -> str | None:
        if isinstance(self.selector, str):
            return self.selector
        if codebase.ordered_paths:
            return codebase.ordered_paths[self.selector % len(codebase.ordered_paths)]
        return None


@dataclasses.dataclass
class CanonicalCodebaseIndex:
    source: CodebaseWithEdits
    sha: Annotated[str, Examples(hashlib.sha1(s).hexdigest() for s in specialized.byte_strings)]

    def find_documents(self, paths: list[str]):
        storage = self.source.ensure()
        with Session() as session:
            return {
                k: [(chunk.path, chunk.index, chunk.hash) for chunk in v]
                for k, v in storage.find_documents(paths, session).items()
            }

    def current_db_repository_info(self) -> DbRepositoryInfo:
        storage = self.source.ensure()
        with Session() as session:
            return (
                session.execute(
                    select(DbRepositoryInfo).filter(DbRepositoryInfo.id == storage.repo_id)
                )
                .scalars()
                .first()
            )

    def current_db_chunks(self) -> list[DbDocumentChunk]:
        storage = self.source.ensure()
        with Session() as session:
            return list(
                session.execute(
                    select(DbDocumentChunk)
                    .filter(
                        DbDocumentChunk.repo_id == storage.repo_id,
                        DbDocumentChunk.namespace.is_(None),
                    )
                    .order_by(DbDocumentChunk.path, DbDocumentChunk.index)
                ).scalars()
            )

    def apply(self):
        storage = self.source.ensure()
        with Session() as session:
            storage.apply_namespace(self.sha, session)
            session.commit()


@dataclasses.dataclass
class FollowupUpdates:
    changes: tuple[FollowupUpdate, ...]
    deletes: list[str] = dataclasses.field(default_factory=list)
    updates: dict[str, list[EmbeddedDocumentChunk]] = dataclasses.field(default_factory=dict)
    new_documents: dict[str, list[EmbeddedDocumentChunk]] = dataclasses.field(default_factory=dict)
    unchanged_paths: set[str] = dataclasses.field(default_factory=set)

    def prepare(self, source: CodebaseWithEdits):
        self.unchanged_paths = set(c.path for c in source.all_chunks)

        for change in self.changes:
            path = change.select(source)
            if path is None:
                continue

            if path in source.documents:
                if path not in self.unchanged_paths:
                    continue
                self.unchanged_paths.remove(path)

            if change.replacement is None:
                self.deletes.append(path)
            else:
                chunks = change.replacement.as_embedded_chunks(path)
                if path in source.documents:
                    self.updates[path] = chunks
                else:
                    self.new_documents[path] = chunks

    def apply(self, source: CodebaseWithEdits):
        storage = source.ensure()
        with Session() as session:
            for path, chunks in self.updates.items():
                storage.replace_documents(chunks, session)
            for path, chunks in self.new_documents.items():
                storage.replace_documents(chunks, session)
            storage.delete_paths(self.deletes, session)
            session.commit()


@parametrize
def test_ensure_codebase_idempotency(
    codebase_one: CodebaseWithEdits, codebase_two: CodebaseWithEdits
):
    @change_watcher
    def count_repo_definitions_in_db():
        with Session() as session:
            return session.execute(select(count(DbRepositoryInfo.id))).scalar_one()

    with count_repo_definitions_in_db as changes:
        assert codebase_one.ensure()
    assert changes.to_value(1)

    with count_repo_definitions_in_db as changes:
        repo_two_id = codebase_two.ensure().repo_id
    assert changes.to_value(2)

    # Idempotent operation
    with count_repo_definitions_in_db as changes:
        assert codebase_two.ensure().repo_id == repo_two_id
    assert not changes


@parametrize
def test_apply_namespace(
    canonical: CanonicalCodebaseIndex,
    unrelated_namespace: CodebaseWithEdits,
):
    unrelated_namespace.apply()
    canonical.source.apply()
    canonical.apply()

    assert canonical.current_db_repository_info().sha == canonical.sha

    assert sorted((c.path, c.index, c.hash) for c in canonical.current_db_chunks()) == sorted(
        (c.path, c.index, c.hash) for c in canonical.source.all_chunks
    )

    assert sorted(
        (c.path, c.index, c.hash) for c in unrelated_namespace.current_db_chunks()
    ) == sorted((c.path, c.index, c.hash) for c in unrelated_namespace.all_chunks)


@parametrize(count=30)
def test_find_documents_after_edits(
    canonical: CanonicalCodebaseIndex,
    followup: FollowupUpdates,
    unrelated_namespace: CodebaseWithEdits,
):
    unrelated_namespace.apply()
    canonical.source.apply()
    canonical.apply()

    followup.prepare(canonical.source)

    assert sometimes(followup.deletes)
    assert sometimes(followup.new_documents)
    assert sometimes(followup.updates)
    assert sometimes(followup.unchanged_paths)

    namespace_tombstone_watcher = change_watcher(lambda: canonical.source.tombstone_paths())
    namespace_chunk_watcher = change_watcher(
        lambda: sorted((c.path, c.index, c.hash) for c in canonical.source.current_db_chunks())
    )
    canonical_chunk_watcher = change_watcher(
        lambda: sorted((c.path, c.index, c.hash) for c in canonical.current_db_chunks())
    )
    unchanged_watcher = change_watcher(
        lambda: canonical.find_documents(list(followup.unchanged_paths))
    )

    with contextlib.ExitStack() as stack:
        unchanged_files_document_changes = stack.enter_context(unchanged_watcher)
        canonical_chunk_changes = stack.enter_context(canonical_chunk_watcher)
        followup.apply(canonical.source)

    assert not unchanged_files_document_changes
    assert not canonical_chunk_changes

    assert canonical.find_documents(followup.deletes) == {}
    assert canonical.find_documents(list(followup.new_documents.keys())) == {
        k: [(chunk.path, chunk.index, chunk.hash) for chunk in v]
        for k, v in followup.new_documents.items()
    }
    assert canonical.find_documents(list(followup.updates.keys())) == {
        k: [(chunk.path, chunk.index, chunk.hash) for chunk in v]
        for k, v in followup.updates.items()
    }

    with contextlib.ExitStack() as stack:
        namespace_changes = stack.enter_context(namespace_chunk_watcher)
        namespace_tombstone_changes = stack.enter_context(namespace_tombstone_watcher)
        canonical.apply()

    if followup.new_documents or followup.updates:
        assert namespace_changes.to_value([])
    if followup.deletes:
        assert namespace_tombstone_changes.to_value([])

    assert sorted(
        (chunk.path, chunk.index, chunk.hash) for chunk in canonical.current_db_chunks()
    ) == sorted(
        (chunk.path, chunk.index, chunk.hash)
        for chunks in (
            *followup.new_documents.values(),
            *followup.updates.values(),
            *(canonical.source.ordered_chunks[path] for path in followup.unchanged_paths),
        )
        for chunk in chunks
    )
