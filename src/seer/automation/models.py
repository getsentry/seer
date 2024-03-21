from typing import List, Literal, Optional
from xml.etree import ElementTree as ET

from pydantic import BaseModel
from pydantic_xml import BaseXmlModel

from seer.automation.agent.models import Usage


class InitializationError(Exception):
    pass


class PromptXmlModel(BaseXmlModel):
    def _pad_with_newlines(self, tree: ET.Element) -> None:
        for elem in tree.iter():
            if elem.text:
                stripped = elem.text.strip("\n")
                if stripped:
                    elem.text = "\n" + stripped + "\n"
            if elem.tail:
                stripped = elem.tail.strip("\n")
                if stripped:
                    elem.tail = "\n" + stripped + "\n"

    def to_prompt_str(self) -> str:
        tree: ET.Element = self.to_xml_tree()

        ET.indent(tree, space="", level=0)

        self._pad_with_newlines(tree)

        return ET.tostring(tree, encoding="unicode")


class Line(BaseModel):
    source_line_no: Optional[int] = None
    target_line_no: Optional[int] = None
    diff_line_no: Optional[int] = None
    value: str
    line_type: Literal[" ", "+", "-"]


class Hunk(BaseModel):
    source_start: int
    source_length: int
    target_start: int
    target_length: int
    section_header: str
    lines: List[Line]


class FilePatch(BaseModel):
    type: Literal["A", "M", "D"]
    path: str
    added: int
    removed: int
    source_file: str
    target_file: str
    hunks: List[Hunk]


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
            if file_contents is not None and file_contents != "":
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
