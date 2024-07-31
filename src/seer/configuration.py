import logging
import os.path
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from seer.dependency_injection import Module

logger = logging.getLogger(__name__)

configuration_module = Module()
configuration_test_module = Module()


class CodebaseStorageType(str, Enum):
    FILESYSTEM = "filesystem"
    GCS = "gcs"


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
    CODEBASE_STORAGE_TYPE: CodebaseStorageType = CodebaseStorageType.FILESYSTEM
    CODEBASE_STORAGE_DIR: ParsePath = os.path.abspath("data/chroma/storage")

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

    CODEBASE_GCS_STORAGE_BUCKET: str = "sentry-ml"
    CODEBASE_GCS_STORAGE_DIR: str = "tmp_jenn/dev/chroma/storage"

    JSON_API_SHARED_SECRETS: ParseList = Field(default_factory=list)
    TORCH_NUM_THREADS: ParseInt = 0
    NO_SENTRY_INTEGRATION: ParseBool = False
    DEV: ParseBool = False

    @property
    def is_production(self) -> bool:
        return not self.DEV

    @property
    def has_sentry_integration(self) -> bool:
        return not self.NO_SENTRY_INTEGRATION

    def do_validation(self):
        if self.is_production:
            assert self.has_sentry_integration, "Sentry integration required for production mode."
            assert self.SENTRY_DSN, "SENTRY_DSN required for production!"

            assert self.LANGFUSE_HOST, "LANGFUSE_HOST required for production!"
            assert self.LANGFUSE_PUBLIC_KEY, "LANGFUSE_PUBLIC_KEY required for production!"
            assert self.LANGFUSE_SECRET_KEY, "LANGFUSE_SECRET_KEY required for production!"

            assert self.GITHUB_APP_ID, "GITHUB_APP_ID required for production!"
            assert self.GITHUB_PRIVATE_KEY, "GITHUB_PRIVATE_KEY required for production!"

            # These are not required for production but is needed to work for customers that don't want PRs to be made.
            if not self.DEV and not self.GITHUB_SENTRY_APP_ID:
                logger.warn("GITHUB_SENTRY_APP_ID is missing in production!")
            if not self.DEV and not self.GITHUB_SENTRY_PRIVATE_KEY:
                logger.warn("GITHUB_SENTRY_PRIVATE_KEY is missing in production!")


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
    base.CODEBASE_STORAGE_DIR = os.path.abspath("data/tests/chroma/storage")
    base.CODEBASE_GCS_STORAGE_DIR = os.path.abspath("chroma-test/data/storage")
    base.DATABASE_URL = base.DATABASE_URL.replace("db", "test-db")
    base.LANGFUSE_HOST = ""
    base.LANGFUSE_PUBLIC_KEY = ""
    base.LANGFUSE_SECRET_KEY = ""

    return base


configuration_module.enable()
