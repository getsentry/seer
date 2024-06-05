import logging
import os
from typing import Collection

import sentry_sdk
from celery import Celery
from flask import Flask
from psycopg import Connection
from sentry_sdk.integrations import Integration
from sqlalchemy.ext.asyncio import create_async_engine

from celery_app.config import CeleryQueues
from seer.db import AsyncSession, Session, db, migrate

logger = logging.getLogger(__name__)


def traces_sampler(sampling_context: dict):
    if "wsgi_environ" in sampling_context:
        path_info = sampling_context["wsgi_environ"].get("PATH_INFO")
        if path_info and path_info.startswith("/health/"):
            return 0.0

    return 1.0


class DisablePreparedStatementConnection(Connection):
    pass


def bootup(
    name: str,
    plugins: Collection[Integration] = (),
    init_migrations=False,
    init_db=True,
    with_async=False,
    async_load_models=False,
) -> Flask:
    grouping_logger = logging.getLogger("grouping")
    grouping_logger.setLevel(logging.INFO)
    grouping_logger.addHandler(logging.StreamHandler())

    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        integrations=[*plugins],
        traces_sampler=traces_sampler,
        profiles_sample_rate=1.0,
        enable_tracing=True,
        traces_sample_rate=1.0,
        send_default_pii=True,
    )
    app = Flask(name)

    uri = os.environ["DATABASE_URL"]
    app.config["SQLALCHEMY_DATABASE_URI"] = uri
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"prepare_threshold": None}}

    from seer.inference_models import start_loading

    if init_db:
        db.init_app(app)
        if init_migrations:
            migrate.init_app(app, db)
        with app.app_context():
            Session.configure(bind=db.engine)
            if with_async:
                AsyncSession.configure(
                    bind=create_async_engine(
                        db.engine.url, connect_args={"prepare_threshold": None}
                    )
                )

    if async_load_models:
        start_loading(async_load_models)

    torch_num_threads = os.environ.get("TORCH_NUM_THREADS")
    if torch_num_threads:
        import torch

        torch.set_num_threads(int(torch_num_threads))

    return app


CELERY_CONFIG = dict(
    broker_url=os.environ.get("CELERY_BROKER_URL"),
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    task_default_queue=CeleryQueues.DEFAULT,
    task_queues={
        CeleryQueues.DEFAULT: {
            "exchange": CeleryQueues.DEFAULT,
            "routing_key": CeleryQueues.DEFAULT,
        },
        CeleryQueues.CUDA: {
            "exchange": CeleryQueues.CUDA,
            "routing_key": CeleryQueues.CUDA,
        },
    },
    result_backend="rpc://",
)


def bootup_celery() -> Celery:
    if not CELERY_CONFIG["broker_url"]:
        logger.warning("CELERY_BROKER_URL not set")

    app = Celery("seer")
    for k, v in CELERY_CONFIG.items():
        setattr(app.conf, k, v)
    return app
