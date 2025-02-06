import textwrap
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, StringConstraints, field_validator
from pydantic_xml import attr, element

from seer.automation.agent.models import Message
from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    TimelineEvent,
)
from seer.automation.autofix.components.solution.models import SolutionTimelineEvent
from seer.automation.autofix.utils import remove_code_backticks
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, Profile, PromptXmlModel
from seer.automation.summarize.issue import IssueSummary


class CodingRequest(BaseComponentRequest):
    event_details: EventDetails
    root_cause: RootCauseAnalysisItem | str
    solution: list[SolutionTimelineEvent] | str
    original_instruction: str | None = None
    root_cause_extra_instruction: str | None = None
    summary: Optional[IssueSummary] = None
    initial_memory: list[Message] = []
    profile: Profile | None = None
    mode: Literal["all", "fix", "test"] = "fix"


class SnippetXml(PromptXmlModel, tag="snippet"):
    file_path: str = attr()
    snippet: Annotated[str, StringConstraints(strip_whitespace=True)]


class TimelineEventXml(PromptXmlModel, tag="step"):
    title: str = element()
    description: str = element()
    relevant_code_file_path: str | None = element(default=None)
    relevant_code_repo_name: str | None = element(default=None)

    @classmethod
    def from_root_cause_context(cls, event: TimelineEvent):
        return cls(
            title=event.title,
            description=event.code_snippet_and_analysis,
            relevant_code_file_path=(
                event.relevant_code_file.file_path if event.relevant_code_file else None
            ),
            relevant_code_repo_name=(
                event.relevant_code_file.repo_name if event.relevant_code_file else None
            ),
        )


class RootCausePlanTaskPromptXml(PromptXmlModel, tag="root_cause", skip_empty=True):
    contexts: list[TimelineEventXml]

    @classmethod
    def from_root_cause(cls, root_cause: RootCauseAnalysisItem):
        return cls(
            contexts=(
                [
                    TimelineEventXml.from_root_cause_context(event)
                    for event in root_cause.root_cause_reproduction
                ]
                if root_cause.root_cause_reproduction
                else []
            )
        )


class PlanStep(BaseModel):
    id: int
    title: str
    text: str


class PlanTaskPromptXml(PromptXmlModel, tag="step"):
    file_path: str = attr()
    repo_name: str = attr()
    type: str = attr()  # This is not a literal in order for pydantic-xml to work
    diff: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    commit_message: Annotated[str, StringConstraints(strip_whitespace=True)] = element()

    @field_validator("diff")
    @classmethod
    def clean_diff(cls, v: str) -> str:
        return remove_code_backticks(v)

    @classmethod
    def get_example(
        cls,
    ):
        return cls(
            file_path="path/to/file.py",
            repo_name="owner/repo",
            type="Either 'file_change', 'file_create', or 'file_delete'",
            description="Describe what you are doing here in detail like you are explaining it to a software engineer.",
            diff=textwrap.dedent(
                """\
                # Here provide the EXACT unified diff of the code change required to accomplish this step.
                # You must prefix lines that are removed with a '-' and lines that are added with a '+'. Context lines around the change are required and must be prefixed with a space.
                # Make sure the diff is complete and the code is EXACTLY matching the files you see.
                # For example:
                --- a/path/to/file.py
                +++ b/path/to/file.py
                @@ -1,3 +1,3 @@
                    return 'fab'
                    y = 2
                    x = 1
                -def foo():
                +def foo():
                    return 'foo'
                    def bar():
                    return 'bar'
                """
            ),
            commit_message="Provide a commit message that describes the change you are making",
        )


class PlanStepsPromptXml(PromptXmlModel, tag="plan_steps"):
    tasks: list[PlanTaskPromptXml]

    @classmethod
    def get_example(cls):
        return cls(
            tasks=[
                PlanTaskPromptXml.get_example(),
                PlanTaskPromptXml.get_example(),
            ]
        )

    def to_model(self):
        return CodingOutput.model_validate(self.model_dump())


class SimpleChangeXml(PromptXmlModel, tag="file_change"):
    file_path: str = attr()
    repo_name: str = attr()
    commit_message: str = element()
    description: str = element()
    unified_diff: str = element()

    def to_plan_task_model(self):
        return PlanTaskPromptXml(
            file_path=self.file_path,
            repo_name=self.repo_name,
            type="file_change",
            diff=self.unified_diff,
            description=self.description,
            commit_message=self.commit_message,
        )


class SimpleChangeOutputXml(PromptXmlModel, tag="output"):
    file_changes: list[SimpleChangeXml]


class CodingOutput(BaseComponentOutput):
    tasks: list[PlanTaskPromptXml]


class FuzzyDiffChunk(BaseModel):
    header: str
    original_chunk: str
    new_chunk: str
    diff_content: str


class FileMissingObj(BaseModel):
    file_path: str
    file_content: str
    diff_chunks: list[FuzzyDiffChunk]
    task: PlanTaskPromptXml
