import os.path
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

from seer.bootup import module, stub_module


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

    DATABASE_URL: str
    CELERY_BROKER_URL: str
    GITHUB_TOKEN: str = ""
    GITHUB_APP_ID: str = ""
    GITHUB_PRIVATE_KEY: str = ""

    CODEBASE_GCS_STORAGE_BUCKET: str = "sentry-ml"
    CODEBASE_GCS_STORAGE_DIR: str = "tmp_jenn/dev/chroma/storage"

    JSON_API_SHARED_SECRETS: ParseList = Field(default_factory=list)
    TORCH_NUM_THREADS: ParseInt = 1
    NO_SENTRY_INTEGRATION: ParseBool = False
    DEV: ParseBool = False

    @property
    def is_production(self) -> bool:
        return not self.DEV

    @property
    def has_sentry_integration(self) -> bool:
        return not self.NO_SENTRY_INTEGRATION

    def model_post_init(self, context: Any):
        if self.is_production:
            assert self.has_sentry_integration, "Sentry integration required for production mode."

        if self.has_sentry_integration:
            assert (
                self.JSON_API_SHARED_SECRETS
            ), "JSON_API_SHARED_SECRETS required for sentry integration."


@module.provider
def load_from_environment(environ: dict[str, str] | None = None) -> AppConfig:
    return AppConfig.model_validate(environ or os.environ)


@stub_module.provider
def provide_test_defaults() -> AppConfig:
    """
    Load defaults into the base app config useful for tests
    """

    base = load_from_environment(
        {
            **os.environ,
            "NO_SENTRY_INTEGRATION": "true",
        }
    )
    base.CODEBASE_STORAGE_DIR = os.path.abspath("data/tests/chroma/storage")
    base.CODEBASE_GCS_STORAGE_DIR = os.path.abspath("chroma-test/data/storage")

    return base