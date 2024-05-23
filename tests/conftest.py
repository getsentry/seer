import os

import pytest
from sqlalchemy import text

from seer.bootup import CELERY_CONFIG, bootup
from seer.db import Session, db
from seer.inference_models import reset_loading_state


@pytest.fixture(autouse=True, scope="session")
def configure_environment():
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL"].replace("db", "test-db")


@pytest.fixture(autouse=True)
def setup_app():
    # Forces the initialization of the database
    reset_loading_state()
    app = bootup(
        __name__,
        init_db=True,
        init_migrations=False,
        with_async=True,
    )

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


import johen
from johen.generators import pydantic, sqlalchemy

johen.global_config["matchers"].extend(
    [
        pydantic.generate_pydantic_instances,
        sqlalchemy.generate_sqlalchemy_instance,
    ]
)
