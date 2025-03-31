from enum import Enum

from seer.automation.component import BaseComponentOutput
from seer.automation.models import FilePatch


class InsightSharingType(str, Enum):
    INSIGHT = "insight"
    FILE_CHANGE = "file_change"


class InsightSharingOutput(BaseComponentOutput):
    insight: str
    justification: str = ""
    change_diff: list[FilePatch] | None = None
    generated_at_memory_index: int = -1
    type: InsightSharingType = InsightSharingType.INSIGHT
