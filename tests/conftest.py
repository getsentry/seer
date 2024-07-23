import os
from typing import Any

import johen
import pytest
from flask import Flask
from johen.generators import pydantic, sqlalchemy
from sqlalchemy import text

from celery_app.config import CeleryConfig
from seer.bootup import bootup, stub_module
from seer.configuration import configuration_test_module
from seer.db import Session, db
from seer.dependency_injection import Module, inject, resolve
from seer.inference_models import reset_loading_state


@pytest.fixture
def test_module() -> Module:
    return stub_module


@pytest.fixture(autouse=True)
def setup_app(test_module: Module):
    with configuration_test_module, test_module:
        reset_loading_state()
        bootup(start_model_loading=False, integrations=[])
        app = resolve(Flask)

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
@inject
# The pytest celery plugin depends on putting the celery config into a pytest fixture, so we inject it
# and return it to place it in fixture namespace.
def celery_config(setup_app: Any, celery_config: CeleryConfig):
    return celery_config


@pytest.fixture(autouse=True)
def reset_environ():
    old_env = os.environ
    os.environ = dict(**old_env)
    try:
        yield
    finally:
        os.environ = old_env


johen.global_config["matchers"].extend(
    [
        pydantic.generate_pydantic_instances,
        sqlalchemy.generate_sqlalchemy_instance,
    ]
)
