import logging
import os
from typing import Protocol

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.logging import LoggingIntegration

from seer.injector import Injector

injector = Injector()


class TracesSampler(Protocol):
    def __call__(self, sampling_context: dict) -> float: ...


@injector.provider
def default_sampler() -> TracesSampler:
    def traces_sampler(sampling_context: dict):
        if "wsgi_environ" in sampling_context:
            path_info = sampling_context["wsgi_environ"].get("PATH_INFO")
            if path_info and path_info.startswith("/health/"):
                return 0.0

        return 1.0

    return traces_sampler


@injector.extension
def default_integrations() -> list[Integration]:
    return [
        LoggingIntegration(
            level=logging.DEBUG,  # Capture debug and above as breadcrumbs
        ),
    ]


@injector.initializer
def initialize_sentry(integrations: list[Integration], traces_sampler: TracesSampler):
    commit_sha = os.environ.get("SEER_VERSION_SHA", "")

    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        integrations=[*integrations],
        traces_sampler=traces_sampler,
        profiles_sample_rate=1.0,
        enable_tracing=True,
        traces_sample_rate=1.0,
        send_default_pii=True,
        release=commit_sha,
        environment="production",
    )
