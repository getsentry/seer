from typing import Literal

from pydantic import BaseModel, ConfigDict

from seer.automation.agent.models import Message
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, Profile
from seer.automation.summarize.issue import IssueSummary


class RelevantCodeFile(BaseModel):
    file_path: str
    repo_name: str


class SolutionTimelineEvent(BaseModel):
    title: str
    code_snippet_and_analysis: str
    relevant_code_file: RelevantCodeFile | None
    is_most_important_event: bool = False
    event_type: Literal["internal_code"] | str = "internal_code"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SolutionPlanStep(BaseModel):
    title: str
    code_snippet_and_analysis: str
    relevant_code_file: RelevantCodeFile | None
    is_most_important: bool


class SolutionRequest(BaseComponentRequest):
    event_details: EventDetails
    root_cause_and_fix: RootCauseAnalysisItem | str
    original_instruction: str | None = None
    summary: IssueSummary | None = None
    initial_memory: list[Message] = []
    profile: Profile | None = None


class SolutionOutput(BaseComponentOutput):
    solution_steps: list[SolutionPlanStep]
    summary: str
