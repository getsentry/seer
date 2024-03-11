import asyncio
import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from seer.bootup import CELERY_CONFIG, bootup
from seer.db import Session, db

create_async_engine(
    os.environ["DATABASE_URL"],
    echo=True,
    pool_recycle=3600,  # Recycle connections every hour
    pool_timeout=10,  # Close idle connections after 10 seconds
)

@pytest.fixture(autouse=True, scope="session")
def configure_environment():
    # disables langsmith
    os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "0"
    os.environ["LANGCHAIN_ENDPOINT"] = "test"
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL"].replace("db", "test-db")


@pytest.fixture(autouse=True)
def manage_db():
    # Forces the initialization of the database
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
