import datetime
from typing import Collection

import sqlalchemy
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


# Initialized in src/app.run
db: SQLAlchemy = SQLAlchemy(model_class=Base)
migrate = Migrate(directory="src/migrations")
Session = sessionmaker()


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
        DateTime, default=datetime.datetime.utcnow, index=True, nullable=False
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
        return max(self.scheduled_for - self.scheduled_from, datetime.timedelta(seconds=1))

    def next_schedule(self, now: datetime.datetime) -> datetime.datetime:
        return now + min((self.last_delay() * 2), datetime.timedelta(hours=1))

    @classmethod
    def schedule_stmt(cls, name: str, payload: dict) -> sqlalchemy.UpdateBase:
        insert_stmt = insert(cls).values(name=name, payload=payload)
        return insert_stmt.on_conflict_do_update(
            index_elements=[cls.name],
            set_={cls.payload: payload},
        )

    @classmethod
    def acquire_work(cls, batch_size: int, now: datetime.datetime):
        with Session() as session:
            items: Collection[ProcessRequest] = (
                session.query(cls)
                .where(cls.scheduled_for < now)
                .order_by(cls.scheduled_for)
                .limit(batch_size)
                .with_for_update()
                .all()
            )

            for item in items:
                item.scheduled_for = item.next_schedule(now)
                item.scheduled_from = now
                session.add(item)

            session.commit()
        return items

    def mark_completed_stmt(self) -> sqlalchemy.UpdateBase:
        return delete(type(self)).where(
            type(self).scheduled_from <= self.scheduled_from and type(self).name == self.name
        )


class DbRepositoryInfo(Base):
    __tablename__ = "repositories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization: Mapped[int] = mapped_column(Integer, nullable=False)
    project: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    external_slug: Mapped[str] = mapped_column(String, nullable=False)
    sha: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (db.UniqueConstraint("organization", "project", "provider", "external_slug"),)


class DbDocumentChunk(Base):
    __tablename__ = "document_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo_id: Mapped[int] = mapped_column(Integer, ForeignKey(DbRepositoryInfo.id), nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str] = mapped_column(String, nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=False)
    for_run_id: Mapped[str] = mapped_column(String(36), nullable=True)
