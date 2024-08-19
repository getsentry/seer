from pydantic import BaseModel

from seer.automation.agent.client import GptClient
from seer.automation.codegen.models import CodegenUnitTestsRequest, CodegenUnitTestsResponse
from seer.dependency_injection import inject, injected


class Step(BaseModel):
    reasoning: str
    justification: str


class IssueSummary(BaseModel):
    reason_step_by_step: list[Step]
    summary_of_issue: str
    affects_what_functionality: str
    known_customer_impact: str
    customer_impact_is_known: bool


@inject
def codegen_noop(request: CodegenUnitTestsRequest, gpt_client: GptClient = injected):
    return CodegenUnitTestsResponse(run_id=-1)
