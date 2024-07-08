import os

import johen
import pytest
from johen.generators import pydantic, sqlalchemy
from sqlalchemy import text

from seer.db import Session, db
from seer.injector import Dependencies, Injector


@pytest.fixture
def test_injectors() -> list[Injector]:
    from seer import app

    return [app.injector]


@pytest.fixture(autouse=True)
def enable_injector(test_injectors: list[Injector]):
    injector = Injector()

    @injector.extension
    def dependencies() -> Dependencies:
        # Database injector is always required
        from seer import db

        return [*test_injectors, db.injector]

    with injector.enable():
        yield


# Swap for 'test-db' so as not to destroy development data.
@pytest.fixture(autouse=True, scope="session")
def configure_environment():
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL"].replace("db", "test-db")


@pytest.fixture(autouse=True)
def prepare_db():
    db.metadata.drop_all(bind=db.engine)
    with Session() as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()
    db.metadata.create_all(bind=db.engine)

    try:
        yield
    finally:
        db.metadata.drop_all(bind=db.engine)


pytest_plugins = (
    "pytest_asyncio",
    "celery.contrib.pytest",
)


@pytest.fixture(scope="session")
def celery_config():
    from celery_app.config import get_celery_config

    return get_celery_config()


@pytest.fixture(autouse=True)
def reset_environ():
    old_env = os.environ
    os.environ = dict(**old_env)
    yield
    os.environ = old_env


johen.global_config["matchers"].extend(
    [
        pydantic.generate_pydantic_instances,
        sqlalchemy.generate_sqlalchemy_instance,
    ]
)
