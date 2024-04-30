import contextlib
import datetime
import json
from typing import Optional

import sqlalchemy
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector  # type: ignore
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


# Initialized in src/app.run
db: SQLAlchemy = SQLAlchemy(model_class=Base)
migrate = Migrate(directory="src/migrations")
Session = sessionmaker(autoflush=False, expire_on_commit=False)
AsyncSession = async_sessionmaker(expire_on_commit=False)


class ProcessRequest(Base):
    """
    Stores durable work that is processed by the async.py worker, in contrast to the best effort queue backed
    celery worker.

    Work should have a unique stable name tied to its semantic.  If an existing item for that work exists, the payload
    can be updated, but do not assume that all payloads will be used.  A payload update is a 'suggestion' for how that
    work could be completed, for instance if it includes something that is time bound (tokens, etc).

    Note that there is no guarantee of single delivery / single processing of any particular request.  It is possible
    that multiple, concurrent attempts to process a request can occur, and ideally each is idempotent with respect to
    important side effects.

    When processing of a request fails, its scheduled_for is updated to an exponentially offset future.  When work is
    'obtained', its scheduled_from is updated as well.  Terminating work is done by deleting the row iff its scheduled
    for is the same as when the work was acquired.
    """

    __tablename__ = "process_request"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True, unique=True, nullable=False)
    scheduled_for: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime(2020, 1, 1), index=True, nullable=False
    )
    scheduled_from: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime(2020, 1, 1), nullable=False
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)

    @property
    def is_terminated(self) -> bool:
        return self.scheduled_for < self.scheduled_from

    def last_delay(self) -> datetime.timedelta:
        return max(self.scheduled_for - self.scheduled_from, datetime.timedelta(minutes=1))

    def next_schedule(self, now: datetime.datetime) -> datetime.datetime:
        return now + min((self.last_delay() * 2), datetime.timedelta(hours=1))

    @classmethod
    def schedule_stmt(
        cls,
        name: str,
        payload: dict | str | bytes | BaseModel,
        when: datetime.datetime,
        expected_duration: datetime.timedelta = datetime.timedelta(seconds=0),  # noqa
    ) -> sqlalchemy.UpdateBase:
        scheduled_from = scheduled_for = when
        # This increases last_delay.  When the item is scheduled, the 'next' schedule will be double this.
        scheduled_from -= expected_duration

        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")

        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)

        insert_stmt = insert(cls).values(
            name=name,
            payload=payload,
            scheduled_for=scheduled_for,
            scheduled_from=scheduled_from,
            created_at=when,
        )

        scheduled_for_update = func.least(insert_stmt.excluded.scheduled_for, cls.scheduled_for)

        return insert_stmt.on_conflict_do_update(
            index_elements=[cls.name],
            set_={
                cls.payload: payload,
                cls.scheduled_from: scheduled_for_update - expected_duration,
                cls.scheduled_for: scheduled_for_update,
                cls.created_at: when,
            },
        )

    @classmethod
    def peek_next_scheduled(
        cls, session: sqlalchemy.orm.Session | None = None
    ) -> datetime.datetime | None:
        with contextlib.ExitStack() as stack:
            if session is None:
                session = stack.enter_context(Session())
            result = session.scalar(select(cls.scheduled_for).order_by(cls.scheduled_for).limit(1))
        return result

    @classmethod
    def acquire_work(
        cls, batch_size: int, now: datetime.datetime, session: sqlalchemy.orm.Session | None = None
    ):
        with contextlib.ExitStack() as stack:
            if session is None:
                session = stack.enter_context(Session())
            items: list[ProcessRequest] = list(
                session.scalars(
                    select(cls)
                    .where(cls.scheduled_for < now)
                    .order_by(cls.scheduled_for)
                    .limit(batch_size)
                    .with_for_update()
                ).all()
            )

            for item in items:
                item.scheduled_for = item.next_schedule(now)
                item.scheduled_from = now
                session.add(item)

            session.commit()

            return items

    def mark_completed_stmt(self) -> sqlalchemy.UpdateBase:
        return delete(type(self)).where(
            type(self).id == self.id, type(self).created_at <= self.created_at
        )


class DbRepositoryInfo(Base):
    __tablename__ = "repositories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization: Mapped[int] = mapped_column(BigInteger, nullable=False)
    project: Mapped[int] = mapped_column(BigInteger, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    external_slug: Mapped[str] = mapped_column(String, nullable=False)
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    default_namespace: Mapped[int] = mapped_column(Integer, nullable=True)
    __table_args__ = (
        UniqueConstraint("organization", "project", "provider", "external_id"),
        Index(
            "ix_repository_organization_project_provider_slug",
            "organization",
            "project",
            "provider",
            "external_id",
        ),
    )


class DbCodebaseNamespace(Base):
    __tablename__ = "codebase_namespaces"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(Integer, ForeignKey(DbRepositoryInfo.id), nullable=False)
    sha: Mapped[str] = mapped_column(String(40), nullable=False)
    tracking_branch: Mapped[str] = mapped_column(String, nullable=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")

    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    accessed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("repo_id", "sha"),
        UniqueConstraint("repo_id", "tracking_branch"),
        Index("ix_codebase_namespace_repo_id_sha", "repo_id", "sha"),
        Index("ix_codebase_namespace_repo_id_tracking_branch", "repo_id", "tracking_branch"),
    )


class DbCodebaseNamespaceMutex(Base):
    __tablename__ = "codebase_namespace_mutex"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    namespace_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DbCodebaseNamespace.id), nullable=False
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )


class DbRunState(Base):
    __tablename__ = "run_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)


class DbGroupingRecord(Base):
    __tablename__ = "grouping_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    project_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=False)
    stacktrace_embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=False)
    stacktrace_hash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index(
            "ix_grouping_records_stacktrace_embedding_hnsw",
            "stacktrace_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_grouping_records_project_id",
            "project_id",
        ),
    )
