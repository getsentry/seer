from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from seer.automation.agent.models import Message
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, Profile, TraceTree
from seer.automation.summarize.issue import IssueSummary


class RelevantCodeFileWithUrl(BaseModel):
    file_path: str
    repo_name: str
    url: str | None = None


class SolutionTimelineEvent(BaseModel):
    title: str
    code_snippet_and_analysis: str | None = None
    relevant_code_file: RelevantCodeFileWithUrl | None = None
    is_most_important_event: bool = False
    timeline_item_type: Literal["internal_code", "human_instruction", "repro_test"] = (
        "internal_code"
    )
    is_active: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def alias_event_type(cls, data):
        if isinstance(data, dict) and "event_type" in data and "timeline_item_type" not in data:
            data["timeline_item_type"] = data.pop("event_type")
        if (
            isinstance(data, dict)
            and "timeline_item_type" in data
            and data["timeline_item_type"] != "internal_code"
            and data["timeline_item_type"] != "human_instruction"
            and data["timeline_item_type"] != "repro_test"
        ):
            data["timeline_item_type"] = "internal_code"
        return data


class RelevantCodeFile(BaseModel):
    file_path: str
    repo_name: str


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
    trace_tree: TraceTree | None = None


class SolutionOutput(BaseComponentOutput):
    solution_steps: list[SolutionPlanStep]
    summary: str
