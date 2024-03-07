import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError, field_validator

from seer.automation.agent.models import Usage


class FileChangeError(Exception):
    pass


class FileChange(BaseModel):
    change_type: Literal["create", "edit", "delete"]
    path: str
    reference_snippet: Optional[str] = None
    new_snippet: Optional[str] = None
    description: Optional[str] = None

    def apply(self, file_contents: str | None) -> str | None:
        if self.change_type == "create":
            if file_contents is not None:
                raise FileChangeError("Cannot create a file that already exists.")
            if self.new_snippet is None:
                raise FileChangeError("New snippet must be provided for creating a file.")
            return self.new_snippet

        if file_contents is None:
            raise FileChangeError("File contents must be provided for non-create operations.")

        if self.change_type == "edit":
            if self.new_snippet is None:
                raise FileChangeError("New snippet must be provided for editing a file.")
            if self.reference_snippet is None:
                raise FileChangeError("Reference snippet must be provided for editing a file.")
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
    abs_path: str
    line_no: Optional[int]
    col_no: Optional[int]
    context: list[tuple[int, str]]
    repo_name: Optional[str] = None
    repo_id: Optional[int] = None
    in_app: bool = False


class Stacktrace(BaseModel):
    frames: list[StacktraceFrame]

    def to_str(self, max_frames: int = 16):
        stack_str = ""
        for frame in reversed(self.frames[-max_frames:]):
            col_no_str = f", column {frame.col_no}" if frame.col_no is not None else ""
            repo_str = f" in repo {frame.repo_name}" if frame.repo_name else ""
            line_no_str = f"[Line {frame.line_no}{col_no_str}]" if frame.line_no is not None else "[Line: Unknown]"
            stack_str += f" {frame.function} in file {frame.filename}{repo_str} {line_no_str} ({'In app' if frame.in_app else 'Not in app'})\n"
            for ctx in frame.context:
                is_suspect_line = ctx[0] == frame.line_no
                stack_str += f"{ctx[1]}{'  <-- SUSPECT LINE' if is_suspect_line else ''}\n"
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
                    abs_path=frame["absPath"],
                    col_no=frame["colNo"],
                    context=frame["context"],
                    in_app=frame["inApp"],
                )
            )

        return Stacktrace(frames=frames)


class IssueDetails(BaseModel):
    id: int
    title: str
    short_id: Optional[str] = None
    events: list[SentryEvent]


class RepoDefinition(BaseModel):
    provider: str
    owner: str
    name: str

    @property
    def full_name(self):
        return f"{self.owner}/{self.name}"

    @field_validator("provider", mode="after")
    @classmethod
    def validate_provider(cls, provider: str):
        cleaned_provider = provider
        if provider.startswith("integrations:"):
            cleaned_provider = provider.split(":")[1]

        if cleaned_provider != "github":
            raise ValueError(f"Provider {cleaned_provider} is not supported.")

        return cleaned_provider

    def __hash__(self):
        return hash((self.provider, self.owner, self.name))


class AutofixUserDetails(BaseModel):
    id: int
    display_name: str


class AutofixRequest(BaseModel):
    organization_id: int
    project_id: int
    repos: list[RepoDefinition]
    issue: IssueDetails
    invoking_user: Optional[AutofixUserDetails] = None

    base_commit_sha: Optional[str] = None
    additional_context: Optional[str] = None
    timeout_secs: Optional[int] = None
    last_updated: Optional[datetime.datetime] = None

    @field_validator("repos", mode="after")
    @classmethod
    def validate_repo_duplicates(cls, repos):
        if isinstance(repos, list):
            # Check for duplicates by comparing lengths after converting to a set
            if len(set(repos)) != len(repos):
                raise ValueError("Duplicate repos detected in the request.")
            return repos

        raise ValueError("Not a list of repos.")


class AutofixOutput(BaseModel):
    title: str
    description: str
    pr_url: str
    pr_number: int
    repo_name: str
    usage: Usage


class AutofixEndpointResponse(BaseModel):
    started: bool


class PullRequestResult(BaseModel):
    pr_number: int
    pr_url: str
    repo: RepoDefinition
