from enum import Enum
from pydantic import BaseModel, field_validator
from pydantic_xml import attr

from seer.automation.models import PromptXmlModel, RepoDefinition


class DocumentPromptXml(PromptXmlModel, tag="document", skip_empty=True):
    path: str = attr()
    repository: str | None = attr(default=None)
    content: str


class BaseDocument(BaseModel):
    path: str
    text: str

    def get_prompt_xml(self, repo_name: str | None) -> DocumentPromptXml:
        return DocumentPromptXml(path=self.path, repository=repo_name, content=self.text)


class Document(BaseDocument):
    language: str


class RepoAccessCheckRequest(BaseModel):
class Provider(str, Enum):
    GITHUB = 'github'
    GITLAB = 'gitlab'
    BITBUCKET = 'bitbucket'


    repo: RepoDefinition


    @field_validator('repo.provider')
    def validate_provider(cls, v):
        # Strip 'integrations:' prefix if present
        provider = v.replace('integrations:', '')
        
        try:
            return Provider(provider.lower())
        except ValueError:
            raise ValueError(f"Provider {provider} is not supported. Supported providers are: {', '.join([p.value for p in Provider])}")


# TODO: Remove this once sentry side is updated
class CodebaseStatusCheckRequest(BaseModel):
    organization_id: int
    project_id: int
    repo: RepoDefinition


class RepoAccessCheckResponse(BaseModel):
    has_access: bool


class CodebaseStatusCheckResponse(BaseModel):
    status: str  # CodebaseIndexStatus, but not using the enum here because it breaks JSON schema generation


class MatchXml(PromptXmlModel, tag="result"):
    path: str = attr()
    context: str


class Match(BaseModel):
    line_number: int
    context: str


class SearchResult(BaseModel):
    relative_path: str
    matches: list[Match]
    score: float