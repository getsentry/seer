from typing import Annotated, Optional

from johen import gen
from johen.examples import Examples
from pydantic import BaseModel, Field, StringConstraints
from pydantic_xml import attr, element

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


class RootCauseAnalysisItem(BaseModel):
    id: int = -1
    title: str
    description: str
    reproduction: str
    likelihood: Annotated[float, Examples(r.uniform(0, 1) for r in gen)] = Field(..., ge=0, le=1)
    actionability: Annotated[float, Examples(r.uniform(0, 1) for r in gen)] = Field(..., ge=0, le=1)
    code_context: Optional[list[RootCauseRelevantContext]] = None


class RootCauseAnalysisRelevantContext(BaseModel):
    snippets: list[RootCauseRelevantContext]


class RootCauseAnalysisItemPrompt(BaseModel):
    title: str
    description: str
    likelihood: float
    actionability: float
    reproduction: str
    relevant_code: Optional[RootCauseAnalysisRelevantContext]

    @classmethod
    def from_model(cls, model: RootCauseAnalysisItem):
        return cls(
            title=model.title,
            likelihood=model.likelihood,
            actionability=model.actionability,
            description=model.description,
            reproduction=model.reproduction,
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
                **self.model_dump(),
                "code_context": (
                    self.relevant_code.model_dump()["snippets"] if self.relevant_code else None
                ),
            }
        )


class MultipleRootCauseAnalysisOutputPrompt(BaseModel):
    causes: list[RootCauseAnalysisItemPrompt]


class RootCauseAnalysisOutputPrompt(BaseModel):
    thoughts: Optional[str]
    potential_root_causes: MultipleRootCauseAnalysisOutputPrompt


class RootCauseAnalysisRequest(BaseComponentRequest):
    event_details: EventDetails
    instruction: Optional[str] = None
    summary: Optional[IssueSummary] = None


class RootCauseAnalysisOutput(BaseComponentOutput):
    causes: list[RootCauseAnalysisItem]
