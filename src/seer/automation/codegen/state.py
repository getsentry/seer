import dataclasses
from typing import cast

from pydantic import BaseModel

from seer.automation.codegen.models import CodegenContinuation
from seer.automation.state import DbState, DbStateRunTypes
from seer.db import DbRunState, Session


@dataclasses.dataclass
class CodegenContinuationState(DbState[CodegenContinuation]):
    @classmethod
    def from_id(
        cls, id: int, model: type[BaseModel], type: DbStateRunTypes = DbStateRunTypes.UNIT_TEST
    ) -> "CodegenContinuationState":
        return cast(CodegenContinuationState, super().from_id(id, model, type=type))

    def set(self, state: CodegenContinuation):
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
