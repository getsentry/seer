from typing import Optional

from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails


class ProblemDiscoveryRequest(BaseComponentRequest):
    event_details: EventDetails

    previous_output: Optional[str] = None
    instruction: Optional[str] = None


class ProblemDiscoveryOutput(BaseComponentOutput):
    description: str
    reasoning: str
    actionability_score: float
