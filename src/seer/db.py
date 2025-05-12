import contextlib
import datetime
import json
from enum import StrEnum
from typing import Any, List, Optional

import sqlalchemy
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector  # type: ignore
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    TIMESTAMP,
    BigInteger,
    Boolean,
    Connection,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Sequence,
    String,
    UniqueConstraint,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


@inject
def initialize_database(
    config: AppConfig = injected,
    app: Flask = injected,
):
    # Get the database URL based on whether we're in migration mode
    database_url = (
        config.DATABASE_MIGRATIONS_URL
        if config.IS_DB_MIGRATION and config.DATABASE_MIGRATIONS_URL
        else config.DATABASE_URL
    )

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"prepare_threshold": None},
        "pool_pre_ping": True,
    }

    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        Session.configure(bind=db.engine)


class Base(DeclarativeBase):
    pass


# Initialized in src/app.run
db: SQLAlchemy = SQLAlchemy(model_class=Base)
migrate = Migrate(directory="src/migrations")
Session = sessionmaker(autoflush=False, expire_on_commit=False)


class TaskStatus(StrEnum):
    NOT_QUEUED = "not_queued"
    PROCESSING = "processing"
    QUEUED = "queued"


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


class DbSeerEvent(Base):
    __tablename__ = "seer_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    event_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )


class DbRunState(Base):
    """
    This is the schema of the run_state table that stores 1 row for every Autofix run.
    The value field is a JSON field that has all the detial information of the state.
    """

    __tablename__ = "run_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    type: Mapped[str] = mapped_column(String, nullable=False, default="autofix")
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    last_triggered_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    pr_id: Mapped[int] = relationship("DbPrIdToAutofixRunIdMapping", cascade="all, delete")

    __table_args__ = (
        Index("ix_run_state_group_id", "group_id"),
        Index("ix_run_state_updated_at", "updated_at"),
        Index("ix_run_state_last_triggered_at", "last_triggered_at"),
    )


class DbRunMemory(Base):
    __tablename__ = "run_memory"
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DbRunState.id, ondelete="CASCADE"), primary_key=True
    )
    value: Mapped[str] = mapped_column(JSON, nullable=False)

    __table_args__ = (Index("ix_run_memory_run_id", "run_id"),)


class DbPrIdToAutofixRunIdMapping(Base):
    __tablename__ = "autofix_pr_id_to_run_id"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    pr_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    run_id: Mapped[int] = mapped_column(
        ForeignKey(DbRunState.id, ondelete="CASCADE"), nullable=False
    )
    __table_args__ = (
        UniqueConstraint("provider", "pr_id", "run_id"),
        Index("ix_autofix_pr_id_to_run_id_provider_pr_id", "provider", "pr_id"),
    )


class DbPrContextToUnitTestGenerationRunIdMapping(Base):
    __tablename__ = "codegen_unit_test_generation_pr_context_to_run_id"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    owner: Mapped[str] = mapped_column(String, nullable=False)
    pr_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    repo: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[int] = mapped_column(
        ForeignKey(DbRunState.id, ondelete="CASCADE"), nullable=False
    )
    iterations = mapped_column(Integer, nullable=False, default=0)
    original_pr_url: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("provider", "pr_id", "repo", "owner", "original_pr_url"),
        Index(
            "ix_unit_test_context_repo_owner_pr_id_pr_url",
            "owner",
            "repo",
            "pr_id",
            "original_pr_url",
        ),
    )


class DbSeerProjectPreference(Base):
    __tablename__ = "seer_project_preferences"
    project_id: Mapped[int] = mapped_column(BigInteger, nullable=False, primary_key=True)
    organization_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    repositories: Mapped[List[dict]] = mapped_column(JSON, nullable=False)

    __table_args__ = (UniqueConstraint("organization_id", "project_id"),)


def create_grouping_partition(target: Any, connection: Connection, **kw: Any) -> None:
    for i in range(100):
        connection.execute(
            text(
                f"""
            CREATE TABLE grouping_records_p{i} PARTITION OF grouping_records
            FOR VALUES WITH (MODULUS 100, REMAINDER {i});
            """
            )
        )


class DbGroupingRecord(Base):
    __tablename__ = "grouping_records"
    __table_args__ = (
        Index(
            "ix_grouping_records_new_stacktrace_embedding_hnsw",
            "stacktrace_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_grouping_records_new_project_id",
            "project_id",
        ),
        UniqueConstraint("project_id", "hash", name="u_project_id_hash_composite"),
        {
            "postgresql_partition_by": "HASH (project_id)",
            "listeners": [("after_create", create_grouping_partition)],
        },
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Sequence("grouping_records_id_seq"),
        primary_key=True,
        server_default=text("nextval('grouping_records_id_seq')"),
    )
    project_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)
    error_type: Mapped[str] = mapped_column(String, nullable=True)
    stacktrace_embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=False)
    hash: Mapped[str] = mapped_column(String(32), nullable=False)


class DbDynamicAlert(Base):
    __tablename__ = "dynamic_alerts"
    __table_args__ = (
        UniqueConstraint("external_alert_id"),
        Index(
            "ix_dynamic_alert_external_alert_id",
            "external_alert_id",
        ),
    )
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    project_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    external_alert_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    anomaly_algo_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    timeseries: Mapped[List["DbDynamicAlertTimeSeries"]] = relationship(
        "DbDynamicAlertTimeSeries",
        back_populates="dynamic_alert",
        cascade="all, delete, delete-orphan",
        passive_deletes=True,
        order_by="DbDynamicAlertTimeSeries.timestamp",
    )
    prophet_predictions: Mapped[List["DbProphetAlertTimeSeries"]] = relationship(
        "DbProphetAlertTimeSeries",
        back_populates="dynamic_alert",
        cascade="all, delete, delete-orphan",
        passive_deletes=True,
        order_by="DbProphetAlertTimeSeries.timestamp",
    )
    data_purge_flag: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, native_enum=False),
        nullable=False,
        default=TaskStatus.NOT_QUEUED,
    )
    last_queued_at: Mapped[Optional[datetime.date]] = mapped_column(
        DateTime, nullable=True, default=datetime.datetime.utcnow
    )


class DbDynamicAlertTimeSeries(Base):
    __tablename__ = "dynamic_alert_time_series"
    __table_args__ = (
        UniqueConstraint("dynamic_alert_id", "timestamp"),
        Index("ix_dynamic_alert_time_series_alert_id_timestamp", "dynamic_alert_id", "timestamp"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dynamic_alert_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DbDynamicAlert.id, ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=False), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    # TODO: Make this model extensible so that other algorithms, in addition to matrix profile, are supported. For now storing it as
    # JSON field that can be extended.
    anomaly_algo_data: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    dynamic_alert = relationship(
        "DbDynamicAlert",
        back_populates="timeseries",
    )

    def __str__(self):
        return f"timstamp: {self.timestamp}, value: {self.value}, anomaly_type:{self.anomaly_type}"


class DbSmokeTest(Base):
    __tablename__ = "smoke_tests"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(128), index=True, unique=True, nullable=False)
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)


class DbIssueSummary(Base):
    __tablename__ = "issue_summary"

    group_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    summary: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC)
    )
    fixability_score: Mapped[float] = mapped_column(Float, nullable=True)
    is_fixable: Mapped[bool] = mapped_column(Boolean, nullable=True)
    fixability_score_version: Mapped[int] = mapped_column(Integer, nullable=True)


class DbDynamicAlertTimeSeriesHistory(Base):
    __tablename__ = "dynamic_alert_time_series_history"
    __table_args__ = (Index("ix_dynamic_alert_time_series_history_timestamp", "timestamp"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC)
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String, nullable=False)
    saved_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC)
    )


class DbProphetAlertTimeSeries(Base):
    __tablename__ = "prophet_alert_time_series"
    __table_args__ = (
        UniqueConstraint("dynamic_alert_id", "timestamp"),
        Index("ix_prophet_alert_time_series_alert_id_timestamp", "dynamic_alert_id", "timestamp"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dynamic_alert_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DbDynamicAlert.id, ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=False), nullable=False)
    yhat: Mapped[float] = mapped_column(Float, nullable=False)
    yhat_lower: Mapped[float] = mapped_column(Float, nullable=False)
    yhat_upper: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC)
    )
    dynamic_alert = relationship(
        "DbDynamicAlert",
        back_populates="prophet_predictions",
    )


class DbProphetAlertTimeSeriesHistory(Base):
    __tablename__ = "prophet_alert_time_series_history"
    __table_args__ = (Index("ix_prophet_alert_time_series_history_timestamp", "timestamp"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    yhat: Mapped[float] = mapped_column(Float, nullable=False)
    yhat_lower: Mapped[float] = mapped_column(Float, nullable=False)
    yhat_upper: Mapped[float] = mapped_column(Float, nullable=False)
    saved_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC)
    )


class DbReviewCommentEmbedding(Base):
    """Store PR review comments with their embeddings for pattern matching per organization"""

    __tablename__ = "review_comments_embedding"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    owner: Mapped[str] = mapped_column(String, nullable=False)
    repo: Mapped[str] = mapped_column(String, nullable=False)
    pr_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    body: Mapped[str] = mapped_column(String, nullable=False)
    is_good_pattern: Mapped[bool] = mapped_column(Boolean, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=False)
    comment_metadata: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.datetime.now(datetime.UTC)
    )

    __table_args__ = (
        UniqueConstraint("provider", "pr_id", "repo", "owner"),
        Index("ix_review_comments_repo_owner_pr", "owner", "repo", "pr_id"),
        Index(
            "ix_review_comments_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_review_comments_is_good_pattern", "is_good_pattern"),
    )
