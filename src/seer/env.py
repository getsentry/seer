import os
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from seer.injector import Injector

injector = Injector()


def parse_int_from_env(data: str) -> int:
    return int(data)


def parse_list_from_env(data: str) -> list[str]:
    return data.split()


def parse_bool_from_env(data: str) -> bool:
    if not data.lower() in ("yes", "true", "t", "y", "1", "on"):
        return False
    return True


ParseInt = Annotated[int, BeforeValidator(parse_int_from_env)]
ParseList = Annotated[list[str], BeforeValidator(parse_list_from_env)]
ParseBool = Annotated[bool, BeforeValidator(parse_bool_from_env)]


class Environment(BaseModel):
    DATABASE_URL: str
    CELERY_BROKER_URL: str
    GITHUB_TOKEN: str = ""
    GITHUB_APP_ID: str = ""
    GITHUB_PRIVATE_KEY: str = ""
    JSON_API_SHARED_SECRETS: ParseList = Field(default_factory=list)
    TORCH_NUM_THREADS: ParseInt = 1
    # Set this to get stubs
    NO_SENTRY_INTEGRATION: ParseBool = False
    DEV: ParseBool = False

    @property
    def PRODUCTION(self) -> bool:
        return not self.DEV

    @property
    def SENTRY_INTEGRATION(self) -> bool:
        return not self.NO_SENTRY_INTEGRATION


@injector.provider
def parse_environment() -> Environment:
    try:
        env = Environment.model_validate(os.environ)
    except ValidationError as e:
        raise e

    if env.PRODUCTION:
        assert env.SENTRY_INTEGRATION, "Sentry Integration required for production"

    if env.SENTRY_INTEGRATION:
        assert (
            env.JSON_API_SHARED_SECRETS
        ), "JSON_API_SHARED_SECRETS is required for sentry integration"

    return env
