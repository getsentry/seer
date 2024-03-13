from typing import Optional

from seer.automation.autofix.models import EventDetails
from seer.automation.component import BaseComponentOutput, BaseComponentRequest


class ProblemDiscoveryRequest(BaseComponentRequest):
    event_details: EventDetails

    previous_output: Optional[str] = None
    additional_context: Optional[str] = None


class ProblemDiscoveryOutput(BaseComponentOutput):
    description: str
    reasoning: str
    actionability_score: float
