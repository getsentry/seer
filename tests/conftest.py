import os

import johen
import pytest
from johen.generators import pydantic, sqlalchemy
from sqlalchemy import text

from seer.bootup import CELERY_CONFIG
from seer.db import Session, db
from seer.inference_models import reset_loading_state


@pytest.fixture(autouse=True, scope="session")
def configure_environment():
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL"].replace("db", "test-db")


@pytest.fixture(autouse=True)
def setup_app():
    from seer.app import app

    reset_loading_state()

    with app.app_context():
        db.metadata.drop_all(bind=db.engine)
        with Session() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
        db.metadata.create_all(bind=db.engine)
    try:
        yield
    finally:
        with app.app_context():
            db.metadata.drop_all(bind=db.engine)


pytest_plugins = (
    "pytest_asyncio",
    "celery.contrib.pytest",
)


@pytest.fixture(scope="session")
def celery_config():
    return CELERY_CONFIG


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
