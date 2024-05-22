from seer.automation.codebase.models import Document
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails


class ExecutorRequest(BaseComponentRequest):
    event_details: EventDetails
    retriever_dump: str | None
    documents: list[Document] = []
    task: str
    repo_name: str


class ExecutorOutput(BaseComponentOutput):
    pass
