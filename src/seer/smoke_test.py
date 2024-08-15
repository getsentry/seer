# A job intended to deep probe the health of the system by running various components in a cheap way
# Executed and checked for success to validate celery configuration e2e.
import logging
import uuid
from datetime import datetime
from typing import Any, Mapping

from pydantic import BaseModel, Field

from celery_app.app import celery_app
from seer.configuration import AppConfig
from seer.db import DbSmokeTest, Session
from seer.dependency_injection import inject, injected
from seer.loading import LoadingResult

logger = logging.getLogger(__name__)


class SmokeRequest(BaseModel):
    smoke_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


@celery_app.task
def smoke_test(*args, payload: Mapping[str, Any], **kwds):
    request = SmokeRequest.model_validate(payload)
    with Session() as session:
        test = session.query(DbSmokeTest).filter(DbSmokeTest.request_id == request.smoke_id).first()
        if test is not None:
            test.completed_at = datetime.utcnow()
            session.add(test)
            session.commit()


@inject
def check_smoke_test(app_config: AppConfig = injected) -> LoadingResult:
    with Session() as session:
        test = (
            session.query(DbSmokeTest)
            .filter(DbSmokeTest.request_id == app_config.smoke_test_id)
            .first()
        )
        if test is None:
            test = DbSmokeTest()
            test.request_id = app_config.smoke_test_id
            test.started_at = datetime.utcnow()
            session.add(test)
            session.commit()
            smoke_test.apply_async(
                (),
                dict(
                    payload=SmokeRequest(smoke_id=app_config.smoke_test_id).model_dump(mode="json")
                ),
            )

    if test.completed_at is None:
        return LoadingResult.LOADING

    return LoadingResult.DONE
