from typing import Optional

from seer.automation.autofix.models import SentryEvent
from seer.automation.component import BaseComponentOutput, BaseComponentRequest


class ProblemDiscoveryRequest(BaseComponentRequest):
    sentry_event: SentryEvent

    previous_output: Optional[str] = None
    additional_context: Optional[str] = None


class ProblemDiscoveryOutput(BaseComponentOutput):
    description: str
    reasoning: str
    actionability_score: float
