from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails


class ExecutorRequest(BaseComponentRequest):
    event_details: EventDetails
    retriever_dump: str | None
    task: str


class ExecutorOutput(BaseComponentOutput):
    pass
