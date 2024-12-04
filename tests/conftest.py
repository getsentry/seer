import logging
import os
from urllib.request import Request

import johen
import pytest
from flask import Flask
from johen.generators import pydantic, sqlalchemy
from pytest_alembic import Config
from sqlalchemy import text

from celery_app.config import CeleryConfig
from seer.app import module
from seer.bootup import bootup, stub_module
from seer.configuration import configuration_test_module
from seer.db import Session, db
from seer.dependency_injection import resolve
from seer.inference_models import reset_loading_state
from seer.rpc import rpc_stub_module

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="session")
def configure_environment():
    os.environ["LANGFUSE_HOST"] = ""  # disable Langfuse logging for tests


@pytest.fixture
def alembic_config():
    return Config.from_raw_config({"file": "/app/src/migrations/alembic.ini"})


@pytest.fixture
def alembic_runner(alembic_config: Config, setup_app):
    import pytest_alembic

    app = resolve(Flask)

    with app.app_context():
        db.metadata.drop_all(bind=db.engine)
        with db.engine.connect() as c:
            c.execute(text("""DROP SCHEMA public CASCADE"""))
            c.execute(text("""CREATE SCHEMA public;"""))
            c.commit()

        with pytest_alembic.runner(config=alembic_config, engine=db.engine) as runner:
            runner.set_revision("base")
            yield runner


@pytest.fixture(autouse=True)
def setup_app():
    with module, configuration_test_module, stub_module, rpc_stub_module:
        reset_loading_state()
        bootup(start_model_loading=False, integrations=[])
        app = resolve(Flask)
        app.testing = True
        # Makes it easier to see stack traces that fail flask endpoint tests.
        app.config["PROPAGATE_EXCEPTIONS"] = True

        # Clean up and recreate the database using the `create_all` rather than invoking migrations over and over
        # is much more efficient in practice.
        with app.app_context():
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
# The pytest celery plugin depends on putting the celery config into a pytest fixture, so we inject it
# and return it to place it in fixture namespace.
def celery_config():
    return resolve(CeleryConfig)


@pytest.fixture(autouse=True)
def reset_environ():
    old_env = os.environ
    os.environ = dict(**old_env)
    try:
        yield
    finally:
        os.environ = old_env


def filter_unrelated_requests(request: Request):
    if request.host.startswith("192"):
        return None
    if "sentry.io" in request.host:
        return None
    return request


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [("authorization", "redacted")],
        "record_mode": "none" if os.environ.get("CI") else "once",
        "before_record_request": filter_unrelated_requests,
        "filter_post_data_parameters": ["client_secret", "refresh_token"],
        "ignore_hosts": ["169.254.169.254"],
    }


johen.global_config["matchers"].extend(
    [
        pydantic.generate_pydantic_instances,
        sqlalchemy.generate_sqlalchemy_instance,
    ]
)
