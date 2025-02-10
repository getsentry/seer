from typing import Literal

from pydantic import BaseModel

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
    timeline_item_type: (
        Literal["internal_code", "external_system", "human_action"] | str
    )  # TODO put back to literal only when not breaking anything
    relevant_code_file: RelevantCodeFile | None
    is_new_event: bool


class SolutionRequest(BaseComponentRequest):
    event_details: EventDetails
    root_cause_and_fix: RootCauseAnalysisItem | str
    original_instruction: str | None = None
    summary: IssueSummary | None = None
    initial_memory: list[Message] = []
    profile: Profile | None = None


class SolutionOutput(BaseComponentOutput):
    modified_timeline: list[SolutionTimelineEvent]
