from typing import Annotated, Optional

from pydantic import BaseModel, Field, StringConstraints
from pydantic_xml import attr, element

from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import EventDetails, PromptXmlModel


class SnippetPromptXml(PromptXmlModel, tag="snippet"):
    file_path: str = attr()
    snippet: Annotated[str, StringConstraints(strip_whitespace=True)]

    @classmethod
    def get_example(cls):
        return cls(
            file_path="path/to/file.py",
            snippet="# This snippet is optional. If there is a direct code fix then include it, if not then don't\ndef foo():\n    return 'bar'\n",
        )


class RootCauseSuggestedFixSnippet(BaseModel):
    file_path: str
    snippet: str


class RootCauseSuggestedFix(BaseModel):
    id: int = -1
    title: str
    description: str
    snippet: Optional[RootCauseSuggestedFixSnippet] = None
    elegance: float


class RootCauseSuggestedFixPromptXml(PromptXmlModel, tag="suggested_fix", skip_empty=True):
    title: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    snippet: Optional[SnippetPromptXml] = None
    elegance: float = attr()

    @classmethod
    def get_example(cls):
        return cls(
            title="Fix the foo() function by returning 'bar'",
            description="This is the ideal fix because... 'bar' is the wrong value because it should be 'baz' instead of 'bar', doing so affects...\nMake sure you use expert judgement and suggest a staff engineer level fix.",
            snippet=SnippetPromptXml.get_example(),
            elegance=0.5,
        )


class RootCauseAnalysisItem(BaseModel):
    id: int = -1
    title: str
    description: str
    likelihood: float = Field(..., ge=0, le=1)
    actionability: float = Field(..., ge=0, le=1)
    suggested_fixes: Optional[list[RootCauseSuggestedFix]] = None


class RootCauseAnalysisSuggestedFixesPromptXml(PromptXmlModel, tag="suggested_fixes"):
    fixes: list[RootCauseSuggestedFixPromptXml]

    @classmethod
    def get_example(cls):
        return cls(fixes=[RootCauseSuggestedFixPromptXml.get_example()])


class RootCauseAnalysisItemPromptXml(PromptXmlModel, tag="potential_cause", skip_empty=True):
    title: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    likelihood: float = attr()
    actionability: float = attr()
    suggested_fixes: Optional[RootCauseAnalysisSuggestedFixesPromptXml] = None

    @classmethod
    def get_example(cls):
        return cls(
            title="The foo() function is returning the wrong value",
            likelihood=0.8,
            actionability=1.0,
            description="The root cause of the issue is that the foo() function is returning the wrong value",
            suggested_fixes=RootCauseAnalysisSuggestedFixesPromptXml.get_example(),
        )

    def to_model(self):
        return RootCauseAnalysisItem.model_validate(
            {
                **self.model_dump(),
                "suggested_fixes": (
                    self.suggested_fixes.model_dump()["fixes"] if self.suggested_fixes else None
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
                    title="All these helper functions seem to be called on the api request POST .../v1/foo, and the request is malformed",
                    likelihood=0.5,
                    actionability=0.1,
                    description="The root cause of the issue is that all these helper functions seem to be called on the api request POST .../v1/foo, and the request is malformed",
                ),
                RootCauseAnalysisItemPromptXml(
                    title="The upstream bar() function sends an incorrect value to foo(), which itself does not have validation, causing this downstream error",
                    likelihood=0.2,
                    actionability=1.0,
                    description="The root cause of the issue is that the upstream bar() function sends an incorrect value to foo(), which itself does not have validation",
                    suggested_fix=RootCauseSuggestedFixPromptXml.get_example(),
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
