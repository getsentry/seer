import dataclasses
from datetime import datetime

from seer.automation.codegen.models import CodegenStatus
from seer.automation.codegen.state import CodegenContinuationState
from seer.automation.models import FileChange


@dataclasses.dataclass
class CodegenEventManager:
    state: CodegenContinuationState

    def mark_running(self):
        with self.state.update() as cur:
            cur.status = CodegenStatus.IN_PROGRESS

    def mark_completed(self):
        with self.state.update() as cur:
            cur.completed_at = datetime.now()
            cur.status = CodegenStatus.COMPLETED

    def add_log(self, message: str):
        pass

    def append_file_change(self, file_change: FileChange):
        with self.state.update() as current_state:
            current_state.file_changes.append(file_change)

    def on_error(
        self, error_msg: str = "Something went wrong", should_completely_error: bool = True
    ):
        with self.state.update() as cur:
            cur.status = CodegenStatus.ERRORED