from typing import Optional

from pydantic import BaseModel

from seer.automation.autofix.components.assessment.models import ProblemDiscoveryOutput
from seer.automation.autofix.models import EventDetails
from seer.automation.component import BaseComponentOutput, BaseComponentRequest


class PlanningRequest(BaseComponentRequest):
    event_details: EventDetails
    problem: ProblemDiscoveryOutput
    instruction: Optional[str] = None


class PlanStep(BaseModel):
    id: int
    title: str
    text: str


class PlanningOutput(BaseComponentOutput):
    title: str
    description: str
    steps: list[PlanStep]
