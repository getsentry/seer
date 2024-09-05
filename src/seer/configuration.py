import logging
import os.path
import uuid
from functools import cached_property
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from seer.dependency_injection import Module

logger = logging.getLogger(__name__)

configuration_module = Module()
configuration_test_module = Module()


def parse_int_from_env(data: str) -> int:
    return int(data)


def parse_list_from_env(data: str) -> list[str]:
    return data.split()


def parse_bool_from_env(data: str) -> bool:
    if not data.lower() in ("yes", "true", "t", "y", "1", "on"):
        return False
    return True


def as_absolute_path(path: str) -> str:
    return os.path.abspath(path)


ParseInt = Annotated[int, BeforeValidator(parse_int_from_env)]
ParseList = Annotated[list[str], BeforeValidator(parse_list_from_env)]
ParseBool = Annotated[bool, BeforeValidator(parse_bool_from_env)]
ParsePath = Annotated[str, BeforeValidator(as_absolute_path)]


class AppConfig(BaseModel):
    SEER_VERSION_SHA: str = ""
    SENTRY_DSN: str = ""

    DATABASE_URL: str
    CELERY_BROKER_URL: str
    GITHUB_TOKEN: str | None = None
    GITHUB_APP_ID: str = ""
    GITHUB_PRIVATE_KEY: str = ""
    GITHUB_SENTRY_APP_ID: str | None = None
    GITHUB_SENTRY_PRIVATE_KEY: str | None = None

    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = ""

    API_PUBLIC_KEY_SECRET_ID: str = ""
    JSON_API_SHARED_SECRETS: ParseList = Field(default_factory=list)
    IGNORE_API_AUTH: ParseBool = False  # Used for both API Tokens and RPC Secrets

    TORCH_NUM_THREADS: ParseInt = 0
    NO_SENTRY_INTEGRATION: ParseBool = False
    DEV: ParseBool = False

    GOOGLE_CLOUD_PROJECT: str = ""
    USE_EU_REGION: ParseBool = False

    SMOKE_CHECK: ParseBool = False

    @cached_property
    def smoke_test_id(self) -> str:
        return str(uuid.uuid4())

    @property
    def is_production(self) -> bool:
        return not self.DEV

    @property
    def has_sentry_integration(self) -> bool:
        return not self.NO_SENTRY_INTEGRATION

    def do_validation(self):
        if not self.IGNORE_API_AUTH:
            assert (
                self.JSON_API_SHARED_SECRETS or self.API_PUBLIC_KEY_SECRET_ID
            ), "JSON_API_SHARED_SECRETS or API_PUBLIC_KEY_SECRET_ID required if IGNORE_API_AUTH is false!"

        if self.is_production:
            # TODO: Set and uncomment this
            # assert (
            #     self.GOOGLE_CLOUD_PROJECT
            # ), "GOOGLE_CLOUD_PROJECT required for production! Why is this not set? Huh?"

            assert self.has_sentry_integration, "Sentry integration required for production mode."
            assert self.SENTRY_DSN, "SENTRY_DSN required for production!"

            # assert self.LANGFUSE_HOST, "LANGFUSE_HOST required for production!"
            # assert self.LANGFUSE_PUBLIC_KEY, "LANGFUSE_PUBLIC_KEY required for production!"
            # assert self.LANGFUSE_SECRET_KEY, "LANGFUSE_SECRET_KEY required for production!"

            assert self.GITHUB_APP_ID, "GITHUB_APP_ID required for production!"
            assert self.GITHUB_PRIVATE_KEY, "GITHUB_PRIVATE_KEY required for production!"

            if not self.JSON_API_SHARED_SECRETS:
                logger.warning("No JSON_API_SHARED_SECRETS was configured for this environment")

            # These are not required for production but is needed to work for customers that don't want PRs to be made.
            if not self.DEV and not self.GITHUB_SENTRY_APP_ID:
                logger.warning("GITHUB_SENTRY_APP_ID is missing in production!")
            if not self.DEV and not self.GITHUB_SENTRY_PRIVATE_KEY:
                logger.warning("GITHUB_SENTRY_PRIVATE_KEY is missing in production!")


@configuration_module.provider
def load_from_environment(environ: dict[str, str] | None = None) -> AppConfig:
    return AppConfig.model_validate(environ or os.environ)


@configuration_test_module.provider
def provide_test_defaults() -> AppConfig:
    """
    Load defaults into the base app config useful for tests
    """

    base = load_from_environment()

    base.NO_SENTRY_INTEGRATION = True
    base.DATABASE_URL = base.DATABASE_URL.replace("db", "test-db")
    base.LANGFUSE_HOST = ""
    base.LANGFUSE_PUBLIC_KEY = ""
    base.LANGFUSE_SECRET_KEY = ""
    base.SMOKE_CHECK = True

    return base


configuration_module.enable()
