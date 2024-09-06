import dataclasses
from typing import cast

from pydantic import BaseModel

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.state import DbState, DbStateRunTypes
from seer.db import DbRunState, Session


@dataclasses.dataclass
class ContinuationState(DbState[AutofixContinuation]):
    @classmethod
    def from_id(cls, id: int, model: type[BaseModel]) -> "ContinuationState":
        return cast(ContinuationState, super().from_id(id, model, type=DbStateRunTypes.AUTOFIX))

    def set(self, state: AutofixContinuation):
        state.mark_updated()

        with Session() as session:
            db_state = DbRunState(
                id=self.id,
                value=state.model_dump(mode="json"),
                updated_at=state.updated_at,
                last_triggered_at=state.last_triggered_at,
            )
            session.merge(db_state)
            session.commit()
