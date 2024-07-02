import logging
import os
import sys
from typing import Collection

import sentry_sdk
import structlog
from celery import Celery
from flask import Flask
from psycopg import Connection
from sentry_sdk.integrations import Integration
from structlog import get_logger

from celery_app.config import CeleryQueues
from seer.db import Session, db, migrate

logger = logging.getLogger(__name__)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]
)


def traces_sampler(sampling_context: dict):
    if "wsgi_environ" in sampling_context:
        path_info = sampling_context["wsgi_environ"].get("PATH_INFO")
        if path_info and path_info.startswith("/health/"):
            return 0.0

    return 1.0


class DisablePreparedStatementConnection(Connection):
    pass


def bootup(
    app: Flask,
    plugins: Collection[Integration] = (),
    init_migrations=False,
    init_db=True,
    async_load_models=False,
) -> Flask:
    grouping_logger = logging.getLogger("grouping")
    grouping_logger.setLevel(logging.INFO)
    grouping_logger.addHandler(StructLogHandler(sys.stdout))

    autofix_logger = logging.getLogger("autofix")
    autofix_logger.setLevel(logging.DEBUG)
    autofix_logger.addHandler(StructLogHandler(sys.stdout))

    commit_sha = os.environ.get("SEER_VERSION_SHA", "")

    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        integrations=[*plugins],
        traces_sampler=traces_sampler,
        profiles_sample_rate=1.0,
        enable_tracing=True,
        traces_sample_rate=1.0,
        send_default_pii=True,
        release=f"seer@{commit_sha}",
        environment="production",
    )

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


throwaways = frozenset(
    (
        "threadName",
        "thread",
        "created",
        "process",
        "processName",
        "args",
        "module",
        "filename",
        "levelno",
        "exc_text",
        "msg",
        "pathname",
        "lineno",
        "funcName",
        "relativeCreated",
        "levelname",
        "msecs",
    )
)


class StructLogHandler(logging.StreamHandler):
    def get_log_kwargs(self, record, logger):
        kwargs = {k: v for k, v in vars(record).items() if k not in throwaways and v is not None}
        kwargs.update({"level": record.levelno, "event": record.msg})

        if record.args:
            # record.args inside of LogRecord.__init__ gets unrolled
            # if it's the shape `({},)`, a single item dictionary.
            # so we need to check for this, and re-wrap it because
            # down the line of structlog, it's expected to be this
            # original shape.
            if isinstance(record.args, (tuple, list)):
                kwargs["positional_args"] = record.args
            else:
                kwargs["positional_args"] = (record.args,)

        return kwargs

    def emit(self, record, logger=None):
        # If anyone wants to use the 'extra' kwarg to provide context within
        # structlog, we have to strip all of the default attributes from
        # a record because the RootLogger will take the 'extra' dictionary
        # and just turn them into attributes.
        try:
            if logger is None:
                logger = get_logger()

            logger.log(**self.get_log_kwargs(record=record, logger=logger))
        except Exception:
            if logging.raiseExceptions:
                raise
