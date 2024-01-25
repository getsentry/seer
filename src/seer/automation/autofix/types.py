from typing import Literal, Optional

from pydantic import BaseModel

from seer.automation.agent.agent import Usage


class FileChange(BaseModel):
    change_type: Literal["create", "edit", "delete"]
    path: str
    reference_snippet: Optional[str] = None
    new_snippet: Optional[str] = None
    description: Optional[str] = None

    def apply(self, file_contents: str | None) -> str | None:
        if self.change_type == "create":
            assert file_contents is None
            assert self.new_snippet is not None
            return self.new_snippet

        assert file_contents is not None

        if self.change_type == "edit":
            assert self.new_snippet is not None
            assert self.reference_snippet is not None
            return file_contents.replace(self.reference_snippet, self.new_snippet)

        # Delete
        if self.reference_snippet is None:
            return None

        return file_contents.replace(self.reference_snippet, "")


class PlanStep(BaseModel):
    id: int
    title: str
    text: str


class PlanningOutput(BaseModel):
    title: str
    description: str
    steps: list[PlanStep]


class ProblemDiscoveryCodeSnippet(BaseModel):
    code: str
    filename: str
    description: Optional[str]


class ProblemDiscoveryOutput(BaseModel):
    description: str
    reasoning: str
    actionability_score: float


class ProblemDiscoveryResult(BaseModel):
    status: Literal["CONTINUE", "CANCELLED"]
    description: str
    reasoning: str


class ProblemDiscoveryRequest(BaseModel):
    message: str
    previous_output: ProblemDiscoveryOutput


class PlanningInput(BaseModel):
    message: Optional[str] = None
    previous_output: Optional[PlanningOutput] = None
    problem: Optional[ProblemDiscoveryOutput] = None


class SentryEvent(BaseModel):
    entries: list[dict]

    def build_stacktrace(self):
        stack_str = ""
        for entry in self.entries:
            if "data" not in entry:
                continue
            if "values" not in entry["data"]:
                continue
            for item in entry["data"]["values"]:
                # stack_str += f"{item['type']}: {item['value']}\n"
                if "stacktrace" not in item:
                    continue
                frames = item["stacktrace"]["frames"][::-1]
                for frame in frames[:4]:
                    stack_str += f" {frame['function']} in {frame['filename']} ({frame['lineNo']}:{frame['colNo']})\n"
                    for ctx in frame["context"]:
                        is_suspect_line = ctx[0] == frame["lineNo"]
                        stack_str += f"{ctx[1]}{'  <--' if is_suspect_line else ''}\n"

                    stack_str += "------\n"

        return stack_str


class IssueDetails(BaseModel):
    id: int
    title: str
    events: list[SentryEvent]


class AutofixRequest(BaseModel):
    issue: IssueDetails
    base_commit_sha: str
    additional_context: Optional[str] = None


class AutofixOutput(BaseModel):
    title: str
    description: str
    pr_url: str
    pr_number: int
    repo_name: str
    usage: Usage


class AutofixEndpointResponse(BaseModel):
    started: bool


### RPC Payloads ###


class ExecutionStep(BaseModel):
    id: int
    description: str
    status: Literal["COMPLETED", "ERROR", "PENDING", "PROCESSING"]


class StepChangePayload(BaseModel):
    issue_id: int
    completed_step: Literal[
        "problem_discovery", "codebase_indexing", "planning", "execution_step", "autofix"
    ]
    status: Literal["COMPLETED", "ERROR", "CANCELLED"]


class ProblemDiscoveryStepChangePayload(StepChangePayload):
    completed_step: Literal["problem_discovery"]
    description: str


class PlanningStepChangePayload(StepChangePayload):
    completed_step: Literal["planning"]
    steps: dict[int, ExecutionStep]


class ExecutionStepChangePayload(StepChangePayload):
    completed_step: Literal["execution_step"]
    execution_id: int


class AutofixResultStepChangePayload(StepChangePayload):
    completed_step: Literal["autofix"]
    fix: AutofixOutput
