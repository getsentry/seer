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
    start_line: int | None = None
    end_line: int | None = None


class RootCauseRelevantContext(BaseModel):
    id: int
    title: str
    description: str
    snippet: Optional[RootCauseRelevantCodeSnippet]


class RootCauseAnalysisRelevantContext(BaseModel):
    snippets: list[RootCauseRelevantContext]

class RootCauseAnalysisItem(BaseModel):
    id: int = -1
    title: str
    description: str
    code_context: Optional[list[RootCauseRelevantContext]] = None

    class Config:
        validate_assignment = True

class RootCauseAnalysisItemPrompt(BaseModel):
    title: str
    description: str
    relevant_code: Optional[RootCauseAnalysisRelevantContext]

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
        return RootCauseAnalysisItem.model_validate(
            {
                **self.model_dump(exclude_none=True),
                "code_context": (
                    self.relevant_code.model_dump()["snippets"] if self.relevant_code else None
                ),
            }
        )


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
