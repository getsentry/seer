import logging
import os
from typing import Collection

import sentry_sdk
from flask import Flask
from sentry_sdk.integrations import Integration
from sqlalchemy.ext.asyncio import create_async_engine

from seer.db import Session, db, migrate
from seer.tasks import AsyncSession


def traces_sampler(sampling_context: dict):
    if "wsgi_environ" in sampling_context:
        path_info = sampling_context["wsgi_environ"].get("PATH_INFO")
        if path_info and path_info.startswith("/health/"):
            return 0.0

    return 1.0


def bootup(
    name: str,
    plugins: Collection[Integration] = (),
    init_migrations=False,
    init_db=True,
    with_async=False,
    eager_load_inference_models=False,
) -> Flask:
    from seer.grouping.grouping import logger as grouping_logger

    grouping_logger.setLevel(logging.INFO)

    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        integrations=[*plugins],
        traces_sampler=traces_sampler,
        profiles_sample_rate=1.0,
        enable_tracing=True,
    )
    app = Flask(name)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URL"]

    from seer.inference_models import cached

    if eager_load_inference_models:
        for item in cached:
            # Preload model
            item()

    if init_db:
        db.init_app(app)
        if init_migrations:
            migrate.init_app(app, db)
        with app.app_context():
            Session.configure(bind=db.engine)
            if with_async:
                AsyncSession.configure(bind=create_async_engine(db.engine.url))

    return app
