from typing import Literal, Optional

from pydantic import BaseModel

from seer.automation.agent.models import Message
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, Profile
from seer.automation.summarize.issue import IssueSummary


class RelevantCodeFile(BaseModel):
    file_path: str
    repo_name: str


class TimelineEvent(BaseModel):
    title: str
    code_snippet_and_analysis: str
    timeline_item_type: (
        Literal["internal_code", "external_system", "human_action"] | str
    )  # TODO put back to literal only when not breaking anything
    relevant_code_file: RelevantCodeFile
    is_most_important_event: bool


class RootCauseAnalysisItem(BaseModel):
    id: int = -1
    root_cause_reproduction: list[TimelineEvent] | None = None
    description: str | None = (
        None  # TODO: this is for backwards compatability with old runs, should remove soon
    )

    def to_markdown_string(self) -> str:
        markdown = "# Root Cause\n\n"

        if self.root_cause_reproduction:
            for event in self.root_cause_reproduction:
                markdown += f"### {event.title}\n"
                markdown += f"{event.code_snippet_and_analysis}\n\n"

        return markdown.strip()


class RootCauseAnalysisItemPrompt(BaseModel):
    root_cause_reproduction: list[TimelineEvent]

    @classmethod
    def from_model(cls, model: RootCauseAnalysisItem):
        return cls(root_cause_reproduction=model.root_cause_reproduction)

    def to_model(self):
        return RootCauseAnalysisItem.model_validate(self.model_dump())


class MultipleRootCauseAnalysisOutputPrompt(BaseModel):
    cause: RootCauseAnalysisItemPrompt


class RootCauseAnalysisOutputPrompt(BaseModel):
    thoughts: Optional[str]
    potential_root_causes: MultipleRootCauseAnalysisOutputPrompt


class RootCauseAnalysisRequest(BaseComponentRequest):
    event_details: EventDetails
    instruction: Optional[str] = None
    summary: Optional[IssueSummary] = None
    initial_memory: list[Message] = []
    profile: Profile | None = None


class RootCauseAnalysisOutput(BaseComponentOutput):
    causes: list[RootCauseAnalysisItem]
    termination_reason: str | None = None
