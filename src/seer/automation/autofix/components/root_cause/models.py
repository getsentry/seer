from typing import Annotated, Optional

from johen import gen
from johen.examples import Examples
from pydantic import BaseModel, Field, StringConstraints
from pydantic_xml import attr, element

from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, PromptXmlModel


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
    id: int = -1
    title: str
    description: str
    snippet: Optional[RootCauseRelevantCodeSnippet] = None


class RootCauseRelevantContextPromptXml(PromptXmlModel, tag="code_snippet", skip_empty=True):
    title: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    snippet: Optional[SnippetPromptXml] = None

    @classmethod
    def get_example(cls):
        return cls(
            title="`foo()` returns the wrong value",
            description="The issue happens because `foo()` always returns `bar`, as seen in this snippet, when it should return `baz`.",
            snippet=SnippetPromptXml.get_example(),
        )


class RootCauseAnalysisItem(BaseModel):
    id: int = -1
    title: str
    description: str
    likelihood: Annotated[float, Examples(r.uniform(0, 1) for r in gen)] = Field(..., ge=0, le=1)
    actionability: Annotated[float, Examples(r.uniform(0, 1) for r in gen)] = Field(..., ge=0, le=1)
    code_context: Optional[list[RootCauseRelevantContext]] = None


class RootCauseAnalysisRelevantContextPromptXml(PromptXmlModel, tag="code_context"):
    snippets: list[RootCauseRelevantContextPromptXml]

    @classmethod
    def get_example(cls):
        return cls(snippets=[RootCauseRelevantContextPromptXml.get_example()])


class RootCauseAnalysisItemPromptXml(PromptXmlModel, tag="potential_cause", skip_empty=True):
    title: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    likelihood: float = attr()
    actionability: float = attr()
    relevant_code: Optional[RootCauseAnalysisRelevantContextPromptXml] = None

    @classmethod
    def get_example(cls):
        return cls(
            title="foo() is returning the wrong value",
            likelihood=0.8,
            actionability=1.0,
            description="The foo() function is returning the wrong value due to a typo in bar().",
            relevant_code=RootCauseAnalysisRelevantContextPromptXml.get_example(),
        )

    @classmethod
    def from_model(cls, model: RootCauseAnalysisItem):
        return cls(
            title=model.title,
            likelihood=model.likelihood,
            actionability=model.actionability,
            description=model.description,
            relevant_code=(
                RootCauseAnalysisRelevantContextPromptXml(
                    snippets=[
                        RootCauseRelevantContextPromptXml(
                            title=snippet.title,
                            description=snippet.description,
                            snippet=(
                                SnippetPromptXml(
                                    file_path=snippet.snippet.file_path,
                                    snippet=snippet.snippet.snippet,
                                    repo_name=snippet.snippet.repo_name,
                                )
                                if snippet.snippet
                                else None
                            ),
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


class MultipleRootCauseAnalysisOutputPromptXml(PromptXmlModel, tag="potential_root_causes"):
    causes: list[RootCauseAnalysisItemPromptXml] = []

    @classmethod
    def get_example(cls):
        return cls(
            causes=[
                RootCauseAnalysisItemPromptXml.get_example(),
                RootCauseAnalysisItemPromptXml(
                    title="bar() sends an incorrect value to foo(), which itself does not have validation",
                    likelihood=0.2,
                    actionability=1.0,
                    description="The upstream bar() function sends an incorrect value to foo(), which itself does not have validation, causing this error downstream.",
                    relevant_code=RootCauseAnalysisRelevantContextPromptXml.get_example(),
                ),
            ]
        )


class RootCauseAnalysisOutputPromptXml(PromptXmlModel, tag="root"):
    thoughts: Optional[str] = element(default=None)
    potential_root_causes: MultipleRootCauseAnalysisOutputPromptXml


class RootCauseAnalysisRequest(BaseComponentRequest):
    event_details: EventDetails

    instruction: Optional[str] = None


class RootCauseAnalysisOutput(BaseComponentOutput):
    causes: list[RootCauseAnalysisItem]
