from seer.automation.codebase.models import BaseDocument
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails


class ExecutorRequest(BaseComponentRequest):
    event_details: EventDetails
    retriever_dump: str | None
    documents: list[BaseDocument] = []
    task: str
    repo_name: str


class ExecutorOutput(BaseComponentOutput):
    pass
