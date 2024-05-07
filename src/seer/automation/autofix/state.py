import dataclasses
from typing import cast

from pydantic import BaseModel

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.state import DbState


@dataclasses.dataclass
class ContinuationState(DbState[AutofixContinuation]):
    @classmethod
    def from_id(cls, id: int, model: type[BaseModel]) -> "ContinuationState":
        return cast(ContinuationState, super().from_id(id, model))

    def set(self, state: AutofixContinuation):
        state.mark_updated()
        super().set(state)
