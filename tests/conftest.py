import asyncio
import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from seer.bootup import bootup
from seer.db import Session, db
from seer.tasks import AsyncSession


@pytest.fixture(autouse=True, scope="session")
def configure_environment():
    os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "0"
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL"].replace("db", "test-db")


@pytest.fixture(autouse=True)
def manage_db():
    # disables langsmith

    # Forces the initialization of the database
    app = bootup(
        __name__,
        init_db=True,
        init_migrations=False,
    )

    with app.app_context():
        with Session() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
        db.metadata.create_all(bind=db.engine)
    try:
        yield
    finally:
        with app.app_context():
            db.metadata.drop_all(bind=db.engine)


# @pytest.fixture(autouse=True)
# def manage_async_errors():
# asyncio.events.get_event_loop().set_exception_handler()

pytest_plugins = ("pytest_asyncio",)
