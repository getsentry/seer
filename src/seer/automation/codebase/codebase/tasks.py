import datetime

import sqlalchemy.orm

from seer.automation.codebase.models import RepositoryInfo
from seer.db import ProcessRequest

schedule_codebase_update_prefix = "schedule-codebase-update:"


def schedule_codebase_update_work(repo_info: RepositoryInfo, session: sqlalchemy.orm.Session):
    session.execute(
        ProcessRequest.schedule_stmt(
            f"{schedule_codebase_update_prefix}{repo_info.id}",
            repo_info.model_dump(),
            datetime.datetime.now(),
            expected_duration=datetime.timedelta(hours=4),
        )
    )
