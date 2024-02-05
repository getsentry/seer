from typing import Literal, Optional

from pydantic import BaseModel

from seer.automation.agent.types import Usage


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


class StacktraceFrame(BaseModel):
    function: str
    filename: str
    line_no: int
    col_no: Optional[int]
    context: list[tuple[int, str]]


class Stacktrace(BaseModel):
    frames: list[StacktraceFrame]

    def to_str(self, max_frames: int = 4):
        stack_str = ""
        for frame in self.frames[:max_frames]:
            stack_str += f" {frame.function} in {frame.filename} ({frame.line_no}:{frame.col_no})\n"
            for ctx in frame.context:
                is_suspect_line = ctx[0] == frame.line_no
                stack_str += f"{ctx[1]}{'  <--' if is_suspect_line else ''}\n"
            stack_str += "------\n"
        return stack_str


class SentryEvent(BaseModel):
    entries: list[dict]

    def get_stacktrace(self):
        exception_entry = next(
            (entry for entry in self.entries if entry["type"] == "exception"),
            None,
        )

        if exception_entry is None:
            return None

        frames: list[StacktraceFrame] = []
        for frame in exception_entry["data"]["values"][0]["stacktrace"]["frames"]:
            frames.append(
                StacktraceFrame(
                    function=frame["function"],
                    filename=frame["filename"],
                    line_no=frame["lineNo"],
                    col_no=frame["colNo"],
                    context=frame["context"],
                )
            )

        return Stacktrace(frames=frames)


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
