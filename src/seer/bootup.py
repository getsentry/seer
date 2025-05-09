import logging


import sentry_sdk
from psycopg import Connection
from sentry_sdk.integrations import Integration
from sentry_sdk.types import Event

from seer.automation.utils import AgentError
from seer.configuration import AppConfig
from seer.db import initialize_database
from seer.dependency_injection import Module, inject, injected
from seer.inference_models import initialize_models

# Make sure this import is here
import seer.logging  # noqa: F401


logger = logging.getLogger(__name__)

module = Module()
stub_module = Module()


def traces_sampler(sampling_context: dict):
    if "wsgi_environ" in sampling_context:
        path_info = sampling_context["wsgi_environ"].get("PATH_INFO")
        if path_info and path_info.startswith("/health/live"):
            return 0.0

    return 1.0


class DisablePreparedStatementConnection(Connection):
    pass


@inject
def bootup(
    *, start_model_loading: bool, integrations: list[Integration], config: AppConfig = injected
):
    initialize_sentry_sdk(integrations)
    with sentry_sdk.metrics.timing(key="seer_bootup_time"):
        config.do_validation()
        initialize_database()
        initialize_models(start_model_loading)


@inject
def initialize_sentry_sdk(integrations: list[Integration], config: AppConfig = injected) -> None:
    def before_send(event: Event, hint: dict) -> Event | None:
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            # exclude errors intended for an AI agent, not Sentry
            if isinstance(exc_value, (AgentError)):
                return None
        return event

    sentry_sdk.init(
        dsn=config.SENTRY_DSN,
        integrations=[*integrations],
        traces_sampler=traces_sampler,
        profiles_sample_rate=config.SENTRY_PROFILES_SAMPLE_RATE,
        send_default_pii=True,
        release=config.SEER_VERSION_SHA,
        environment=config.SENTRY_ENVIRONMENT,
        before_send=before_send,
        _experiments={
            "continuous_profiling_auto_start": True,
            "enable_logs": True,
        },
    )

    if config.SENTRY_REGION:
        sentry_sdk.set_tag("sentry_region", config.SENTRY_REGION)


module.enable()
