from typing import Literal, Optional

from pydantic import BaseModel

from ..agent.agent import Usage


class FileChange(BaseModel):
    change_type: Literal["edit", "delete"]
    path: str
    original_contents: Optional[str]
    contents: Optional[str]
    description: Optional[str]


class AutofixInput(BaseModel):
    additional_context: Optional[str]


class PlanningOutput(BaseModel):
    title: str
    description: str
    plan: str
    usage: Usage


class AutofixOutput(PlanningOutput):
    changes: list[FileChange]


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
    id: str
    title: str
    events: list[SentryEvent]
