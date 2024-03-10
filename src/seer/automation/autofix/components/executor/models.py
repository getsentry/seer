from seer.automation.autofix.models import SentryEvent
from seer.automation.component import BaseComponentOutput, BaseComponentRequest


class ExecutorRequest(BaseComponentRequest):
    sentry_event: SentryEvent
    retriever_dump: str | None
    task: str


class ExecutorOutput(BaseComponentOutput):
    pass
