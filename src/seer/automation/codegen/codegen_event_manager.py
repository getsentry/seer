import dataclasses
from datetime import datetime

from seer.automation.autofix.components.insight_sharing.models import InsightSharingOutput
from seer.automation.codegen.models import CodegenStatus, StaticAnalysisSuggestion
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

    def mark_completed_and_extend_static_analysis_suggestions(
        self, static_analysis_suggestions: list[StaticAnalysisSuggestion]
    ):
        with self.state.update() as cur:
            cur.static_analysis_suggestions.extend(static_analysis_suggestions)
            cur.completed_at = datetime.now()
            cur.status = CodegenStatus.COMPLETED

    def on_error(
        self, error_msg: str = "Something went wrong", should_completely_error: bool = True
    ):
        with self.state.update() as cur:
            cur.status = CodegenStatus.ERRORED

    def send_insight(self, insight_card: InsightSharingOutput):
        """
        Do nothing for now, this is only used for autofix
        """
        pass
