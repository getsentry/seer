from typing import Annotated, Optional, Union

from pydantic import BaseModel, StringConstraints
from pydantic_xml import attr, element

from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, PromptXmlModel


class PlanningRequest(BaseComponentRequest):
    event_details: EventDetails
    root_cause_and_fix: RootCauseAnalysisItem | str
    instruction: Optional[str] = None


class SnippetXml(PromptXmlModel, tag="snippet"):
    file_path: str = attr()
    snippet: Annotated[str, StringConstraints(strip_whitespace=True)]


class RootCausePlanTaskPromptXml(PromptXmlModel, tag="task", skip_empty=True):
    title: str = element()
    description: str = element()
    fix_title: Optional[str] = element()
    fix_description: Optional[str] = element()
    fix_snippet: Optional[SnippetXml]

    @classmethod
    def from_root_cause(cls, root_cause: RootCauseAnalysisItem):
        return cls(
            title=root_cause.title,
            description=root_cause.description,
            fix_title=root_cause.suggested_fixes[0].title if root_cause.suggested_fixes else None,
            fix_description=(
                root_cause.suggested_fixes[0].description if root_cause.suggested_fixes else None
            ),
            fix_snippet=(
                SnippetXml(
                    file_path=root_cause.suggested_fixes[0].snippet.file_path,
                    snippet=root_cause.suggested_fixes[0].snippet.snippet,
                )
                if root_cause.suggested_fixes and root_cause.suggested_fixes[0].snippet
                else None
            ),
        )


class PlanStep(BaseModel):
    id: int
    title: str
    text: str


class ReplaceCodePromptXml(PromptXmlModel, tag="code_change"):
    file_path: str = attr()
    repo_name: str = attr()
    reference_snippet: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    new_snippet: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    new_imports: Optional[Annotated[str, StringConstraints(strip_whitespace=True)]] = element(
        default=None
    )
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    commit_message: Annotated[str, StringConstraints(strip_whitespace=True)] = element()

    @classmethod
    def get_example(cls):
        return cls(
            file_path="path/to/file.py",
            repo_name="owner/repo",
            description="Describe what you are doing here in detail like you are explaining it to a software engineer.",
            reference_snippet="This is the reference snippet, use this to find the code to replace",
            new_snippet="This is the new snippet, this can be an empty opening/closing tag if you are deleting code",
            new_imports="Optional, import statements that need to be added to the TOP of the file",
            commit_message="Fix the foo() function by returning 'bar'",
        )


class CreateFilePromptXml(PromptXmlModel, tag="create_file"):
    file_path: str = attr()
    repo_name: str = attr()
    snippet: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    commit_message: Annotated[str, StringConstraints(strip_whitespace=True)] = element()

    @classmethod
    def get_example(cls):
        return cls(
            file_path="path/to/file.py",
            repo_name="owner/repo",
            description="Describe what you are doing here in detail like you are explaining it to a software engineer.",
            snippet="# This is the new file content",
            commit_message="Create the foo() function that returns 'bar'",
        )


class PlanStepsPromptXml(PromptXmlModel, tag="plan_steps"):
    tasks: list[Union[CreateFilePromptXml, ReplaceCodePromptXml]]

    @classmethod
    def get_example(cls):
        return cls(
            tasks=[
                ReplaceCodePromptXml.get_example(),
                CreateFilePromptXml.get_example(),
            ]
        )


class PlanningOutputPromptXml(PromptXmlModel, tag="planning_output"):
    thoughts: Optional[str] = element(default=None)
    content: Optional[str] = None
    # Order matters here, don't move plan_steps before thoughts
    plan_steps: PlanStepsPromptXml

    def to_model(self):
        return PlanningOutput.model_validate(self.plan_steps.model_dump())


class PlanningOutput(BaseComponentOutput):
    tasks: list[Union[CreateFilePromptXml, ReplaceCodePromptXml]]
