from typing import Annotated, Optional

from pydantic import BaseModel, StringConstraints
from pydantic_xml import attr, element

from seer.automation.autofix.models import EventDetails
from seer.automation.component import BaseComponentOutput, BaseComponentRequest
from seer.automation.models import PromptXmlModel


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
    title: str
    description: str
    snippet: Optional[RootCauseSuggestedFixSnippet] = None


class RootCauseSuggestedFixPromptXml(PromptXmlModel, tag="suggested_fix", skip_empty=True):
    title: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    snippet: Optional[SnippetPromptXml] = None

    @classmethod
    def get_example(cls):
        return cls(
            title="Fix the foo() function by returning 'bar'",
            description="This is the ideal fix because... 'bar' is the wrong value because it should be 'baz' instead of 'bar', doing so affects...",
            snippet=SnippetPromptXml.get_example(),
        )


class RootCauseAnalysisItem(BaseModel):
    title: str
    description: str
    likelihood: float
    actionability: float
    suggested_fix: Optional[RootCauseSuggestedFix] = None


class RootCauseAnalysisOutputPromptXml(PromptXmlModel, tag="potential_cause", skip_empty=True):
    title: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    description: Annotated[str, StringConstraints(strip_whitespace=True)] = element()
    likelihood: float = attr()
    actionability: float = attr()
    suggested_fix: Optional[RootCauseSuggestedFixPromptXml] = None

    @classmethod
    def get_example(cls):
        return cls(
            title="The foo() function is returning the wrong value",
            likelihood=0.8,
            actionability=1.0,
            description="The root cause of the issue is that the foo() function is returning the wrong value",
            suggested_fix=RootCauseSuggestedFixPromptXml.get_example(),
        )

    def to_model(self):
        return RootCauseAnalysisItem.model_validate(self.model_dump())


class MultipleRootCauseAnalysisOutputPromptXml(PromptXmlModel, tag="potential_root_causes"):
    causes: list[RootCauseAnalysisOutputPromptXml]

    @classmethod
    def get_example(cls):
        return cls(
            causes=[
                RootCauseAnalysisOutputPromptXml.get_example(),
                RootCauseAnalysisOutputPromptXml(
                    title="All these helper functions seem to be called on the api request POST .../v1/foo, and the request is malformed",
                    likelihood=0.5,
                    actionability=0.1,
                    description="The root cause of the issue is that all these helper functions seem to be called on the api request POST .../v1/foo, and the request is malformed",
                ),
                RootCauseAnalysisOutputPromptXml(
                    title="The upstream bar() function sends an incorrect value to foo(), which itself does not have validation, causing this downstream error",
                    likelihood=0.2,
                    actionability=1.0,
                    description="The root cause of the issue is that the upstream bar() function sends an incorrect value to foo(), which itself does not have validation",
                    suggested_fix=RootCauseSuggestedFixPromptXml.get_example(),
                ),
            ]
        )


class RootCauseAnalysisRequest(BaseComponentRequest):
    event_details: EventDetails

    instruction: Optional[str] = None


class RootCauseAnalysisOutput(BaseComponentOutput):
    causes: list[RootCauseAnalysisItem]
