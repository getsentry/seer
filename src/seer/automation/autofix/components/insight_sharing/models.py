from seer.automation.component import BaseComponentOutput


class InsightSharingOutput(BaseComponentOutput):
    insight: str
    justification: str = ""
    generated_at_memory_index: int = -1
