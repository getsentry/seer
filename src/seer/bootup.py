import logging
import os

import sentry_sdk
from flask import Flask
from psycopg import Connection
from sentry_sdk.integrations import Integration
from sentry_sdk.types import Event

from seer.automation.utils import AgentError
from seer.configuration import AppConfig
from seer.db import Session, db, migrate
from seer.dependency_injection import Module, inject, injected
from seer.logging import initialize_logs

logger = logging.getLogger(__name__)

module = Module()
stub_module = Module()


@module.provider
def base_app() -> Flask:
    return Flask("seer.app")


def traces_sampler(sampling_context: dict):
    if "wsgi_environ" in sampling_context:
        path_info = sampling_context["wsgi_environ"].get("PATH_INFO")
        if path_info and path_info.startswith("/health/"):
            return 0.0

    return 1.0


class DisablePreparedStatementConnection(Connection):
    pass


@inject
def bootup(
    init_migrations=False,
    init_db=True,
    async_load_models=False,
    config: AppConfig = injected,
    app: Flask = injected,
):
    initialize_logs()
    initialize_sentry_sdk()

    app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URL
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


@inject
def initialize_sentry_sdk(
    integrations: list[Integration] = injected, config: AppConfig = injected
) -> None:
    def before_send(event: Event, hint: dict) -> Event:
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            # exclude errors intended for an AI agent, not Sentry
            if isinstance(exc_value, (AgentError)):
                return None
        return event

    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        integrations=[*integrations],
        traces_sampler=traces_sampler,
        profiles_sample_rate=1.0,
        enable_tracing=True,
        traces_sample_rate=1.0,
        send_default_pii=True,
        release=config.SEER_VERSION_SHA,
        environment="production",
        before_send=before_send,
    )


module.enable()
