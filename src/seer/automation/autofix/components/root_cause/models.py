"""
Root cause analysis models.

Note: Prior versions of these models included unit_test and reproduction fields.
These have been removed in favor of a simplified model structure focusing on
essential fields only. This change was made to reduce complexity and prevent
validation issues with optional fields.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, StringConstraints, field_validator
from pydantic_xml import attr

from seer.automation.agent.models import Message
from seer.automation.autofix.utils import remove_code_backticks
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, PromptXmlModel
from seer.automation.summarize.issue import IssueSummary


class SnippetPromptXml(PromptXmlModel, tag="code"):
    file_path: str = attr()
    repo_name: Optional[str] = attr()
    snippet: Annotated[str, StringConstraints(strip_whitespace=True)]

    @classmethod
    def get_example(cls):
        return cls(
            file_path="path/to/file.py",
            repo_name="owner/repo",
            snippet="def foo():\n    return 'bar'\n",
        )


class RootCauseRelevantCodeSnippet(BaseModel):
    file_path: str
    repo_name: Optional[str]
    snippet: str


class RootCauseRelevantContext(BaseModel):
    id: int
    title: str
    description: str
    snippet: Optional[RootCauseRelevantCodeSnippet]


class RootCauseAnalysisRelevantContext(BaseModel):
    snippets: list[RootCauseRelevantContext]


class UnitTestSnippetPrompt(BaseModel):
    file_path: str
    code_snippet: str
    description: str

    @field_validator("code_snippet")
    @classmethod
    def clean_code_snippet(cls, v: str) -> str:
        return remove_code_backticks(v)


class UnitTestSnippet(BaseModel):
    file_path: str
    snippet: str
    description: str


class RootCauseAnalysisItem(BaseModel):
    id: int = -1
    title: str
    description: str
    code_context: Optional[list[RootCauseRelevantContext]] = None
    
    model_config = {
        "extra": "forbid",  # Prevent any extra fields from being included
        "validate_assignment": True  # Ensure validation on field assignment
    }

    def to_markdown_string(self) -> str:
        markdown = f"# {self.title}\n\n"
        markdown += f"## Description\n{self.description}\n\n" if self.description else ""

        if self.code_context:
            markdown += "## Relevant Code Context\n\n"
            for context in self.code_context:
                markdown += f"### {context.title}\n"
                markdown += f"{context.description}\n\n"
                if context.snippet:
                    markdown += f"**File:** {context.snippet.file_path}\n"
                    if context.snippet.repo_name:
                        markdown += f"**Repository:** {context.snippet.repo_name}\n"
                    markdown += "```\n"
                    markdown += f"{context.snippet.snippet}\n"
                    markdown += "```\n\n"

        return markdown.strip()


class RootCauseAnalysisItemPrompt(BaseModel):
    title: str
    description: str
    relevant_code: Optional[RootCauseAnalysisRelevantContext]
    
    model_config = {
        "extra": "forbid",  # Prevent any extra fields from being included
        "validate_assignment": True  # Ensure validation on field assignment
    }

    @classmethod
    def from_model(cls, model: RootCauseAnalysisItem):
        return cls(
            title=model.title,
            description=model.description,
            relevant_code=(
                RootCauseAnalysisRelevantContext(
                    snippets=[
                        RootCauseRelevantContext(
                            id=snippet.id,
                            title=snippet.title,
                            description=snippet.description,
                            snippet=snippet.snippet,
                        )
                        for snippet in model.code_context
                    ]
                )
                if model.code_context
                else None
            ),
        )

    def to_model(self):
        # Create a clean dictionary with only the required fields
        model_data = {
            "title": self.title,
            "description": self.description,
            "code_context": (
                "code_context": (
                    self.relevant_code.model_dump()["snippets"] if self.relevant_code else None
                )
        }
        return RootCauseAnalysisItem(**model_data)
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


class RootCauseAnalysisOutput(BaseComponentOutput):
    causes: list[RootCauseAnalysisItem]
    termination_reason: str | None = None
