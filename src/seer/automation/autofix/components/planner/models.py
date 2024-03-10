from typing import Optional

from pydantic import BaseModel

from seer.automation.autofix.components.assessment.models import ProblemDiscoveryOutput
from seer.automation.autofix.models import SentryEvent
from seer.automation.component import BaseComponentOutput, BaseComponentRequest


class PlanningRequest(BaseComponentRequest):
    sentry_event: SentryEvent
    problem: ProblemDiscoveryOutput
    additional_context: Optional[str] = None


class PlanStep(BaseModel):
    id: int
    title: str
    text: str


class PlanningOutput(BaseComponentOutput):
    title: str
    description: str
    steps: list[PlanStep]
