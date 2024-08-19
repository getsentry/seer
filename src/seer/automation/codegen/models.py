from pydantic import BaseModel


class CodegenUnitTestsRequest(BaseModel):
    repo_external_id: str  # The Github repo id
    pr_id: int  # The PR number


class CodegenUnitTestsResponse(BaseModel):
    run_id: int
