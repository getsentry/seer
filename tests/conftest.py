import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from seer.db import Session, db
from seer.tasks import AsyncSession


@pytest.fixture(autouse=True)
def manage_db():
    # Forces the initialization of the database
    from seer.app import app

    with app.app_context():
        Session.configure(bind=db.engine)
        AsyncSession.configure(bind=create_async_engine(db.engine.url))  # type: ignore
        with Session() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.metadata.create_all(bind=db.engine)
    try:
        yield
    finally:
        with app.app_context():
            db.metadata.drop_all(bind=db.engine)
