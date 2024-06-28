import abc
import dataclasses

from seer.automation.models import FileChange
from seer.automation.state import LocalMemoryState, State


@dataclasses.dataclass
class CodebaseStateManager(abc.ABC):
    repo_external_id: str
    state: State

    @abc.abstractmethod
    def get_file_changes(self) -> list[FileChange]:
        pass


class DummyCodebaseStateManager(CodebaseStateManager):
    def __init__(self):
        super().__init__("dummy", LocalMemoryState({"file_changes": []}))

    def get_file_changes(self) -> list[FileChange]:
        return self.state.get()["file_changes"]
