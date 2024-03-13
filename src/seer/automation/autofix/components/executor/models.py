from seer.automation.autofix.models import EventDetails
from seer.automation.component import BaseComponentOutput, BaseComponentRequest


class ExecutorRequest(BaseComponentRequest):
    event_details: EventDetails
    retriever_dump: str | None
    task: str


class ExecutorOutput(BaseComponentOutput):
    pass
