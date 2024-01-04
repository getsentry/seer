# Read the documents from the local directory, convert into nodes
import difflib
import json
import logging
import os
import re
from typing import List
from xml.etree import ElementTree as ET

import requests
import torch
from github.Comparison import Comparison
from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding
from llama_index.response.schema import RESPONSE_TYPE
from sentence_transformers import SentenceTransformer

from .types import FileChange

logger = logging.getLogger(__name__)

sentry_auth_token = os.getenv("SENTRY_AUTH_TOKEN")


def diff_to_str(diff: Comparison) -> str:
    changes = ""
    for file in diff.files:
        if file.filename and file.patch:
            changes += file.filename + "\n"
            changes += file.patch
            changes += "\n---\n"

    return changes


def get_issue_url(issue_id: str | int) -> str:
    return f"https://sentry.io/organizations/sentry/issues/{issue_id}/"


def compute_similarity(text1: str, text2: str, ignore_whitespace=True) -> float:
    """
    This function computes the similarity between two pieces of text.

    Parameters:
    text1 (str): The first piece of text.
    text2 (str): The second piece of text.
    ignore_whitespace (bool): If True, ignores whitespace when comparing the two pieces of text.

    Returns:
    float: The similarity ratio between the two pieces of text.
    """
    if ignore_whitespace:
        text1 = re.sub(r"\s+", "", text1)
        text2 = re.sub(r"\s+", "", text2)

    return difflib.SequenceMatcher(None, text1, text2).ratio()


def compute_code_similarity(text1: str, text2: str, ignore_whitespace=True) -> float:
    """
    This function computes the similarity between two pieces of code.

    Parameters:
    text1 (str): The first piece of code.
    text2 (str): The second piece of code.
    ignore_whitespace (bool): If True, ignores whitespace when comparing the two pieces of code.

    The function also handles ellipsis cases ('... ', '// ...', '# ...', '/* ... */') to ensure that they do not trigger on ...expansion in js code.
    If an ellipsis is found and the similarity is less than 0.99, the line is skipped in the similarity computation.

    Returns:
    float: The similarity ratio between the two pieces of code.
    """
    ellipsis_cases = [
        "... ",
        "// ...",
        "# ...",
        "/* ... */",
    ]  # make sure that it does not trigger on ...expansion in js code

    lines1 = text1.split("\n")
    lines2 = text2.split("\n")
    total_similarity = 0
    count = 0

    ellipsis_found = False
    for line1, line2 in zip(lines1, lines2):
        similarity = compute_similarity(line1, line2, ignore_whitespace=ignore_whitespace)

        if not ellipsis_found:
            ellipsis_found = any(s in line1 for s in ellipsis_cases)

        if ellipsis_found and similarity < 0.99:
            continue

        ellipsis_found = False

        total_similarity += similarity
        count += 1

    if count == 0:
        return 0
    else:
        return total_similarity / count


def get_last_non_empty_line(text: str) -> str:
    """
    This function returns the last non-empty line in a piece of text.

    Parameters:
    text (str): A string containing a piece of text.

    Returns:
    str: The last non-empty line in the piece of text.
    """
    lines = text.split("\n")
    for line in reversed(lines):
        if line.strip() != "":
            return line
    return ""


def get_diff(changes: list[FileChange]):
    diffs = []
    for change in changes:
        if change.change_type == "edit":
            diff = f"diff --git a/{change.path} b/{change.path}\n"
            diff += f"--- a/{change.path}\n"
            diff += f"+++ b/{change.path}\n"
            original_lines = (
                change.original_contents.split("\n") if change.original_contents else ""
            )
            new_lines = change.contents.split("\n")
            diff_body = ""
            for i, (orig_line, new_line) in enumerate(zip(original_lines, new_lines)):
                if orig_line != new_line:
                    diff_body += f"-{orig_line}\n+{new_line}\n"
            diff += diff_body
            diffs.append(diff)
        elif change.change_type == "delete":
            diff = f"diff --git a/{change.path} b/{change.path}\n"
            diff += f"deleted file mode 100644\n"
            diff += f"--- a/{change.path}\n"
            diff += f"+++ /dev/null\n"
            diffs.append(diff)
    return "\n".join(diffs)


def find_original_snippet(
    snippet: str, file_contents: str, threshold=0.99, verbose=False
) -> str | None:
    """
    This function finds the original snippet of code in a file given a snippet and the file contents.

    Parameters:
    snippet (str): A string containing a snippet of code.
    file_contents (str): A string containing the contents of a file.

    The function works by splitting the snippet and the file contents into lines and comparing them line by line.
    It uses the compute_similarity function to find the first line of the snippet in the file.
    It then continues comparing the following lines, handling ellipsis cases, until it finds a discrepancy or reaches the end of the snippet.
    If the last line of the snippet is not at least 99% similar to the corresponding line in the file, it returns None.
    Otherwise, it returns the original snippet from the file.

    Returns:
    str: The original snippet from the file, or None if the snippet could not be found.
    """
    if verbose:
        print(f"Finding original snippet in file with {len(file_contents)} characters")
    snippet_lines = snippet.split("\n")
    file_lines = file_contents.split("\n")

    first_line = snippet_lines[0].strip()
    while first_line == "":
        snippet_lines = snippet_lines[1:]
        if len(snippet_lines) == 0:
            return None
        first_line = snippet_lines[0].strip()

    snippet_start = None
    for i, file_line in enumerate(file_lines):
        if compute_similarity(first_line, file_line) > threshold:
            snippet_start = i
            break

    if snippet_start is None:
        return None

    ellipsis_cases = ["... ", "// ...", "# ...", "/* ... */"]
    ellipsis_found = False
    snippet_index = 0
    file_line_index = snippet_start
    while snippet_index < len(snippet_lines) and file_line_index < len(file_lines):
        snippet_line = snippet_lines[snippet_index]

        if not ellipsis_found:
            ellipsis_found = any(s in snippet_line for s in ellipsis_cases)
            if ellipsis_found:
                snippet_index += 1
                if snippet_index >= len(snippet_lines):
                    break
                snippet_line = snippet_lines[snippet_index]

        file_line = file_lines[file_line_index]

        if snippet_line.strip() == "":
            snippet_index += 1
            continue
        if file_line.strip() == "":
            file_line_index += 1
            continue

        similarity = compute_similarity(snippet_line, file_line)

        if ellipsis_found and similarity < threshold:
            file_line_index += 1
        else:
            if similarity < threshold:
                if verbose:
                    print(f"Similarity is less than 99%: {similarity}")
                    print(f"Snippet: {snippet_line}")
                    print(f"File: {file_line}")

                return None
            ellipsis_found = False
            snippet_index += 1
            file_line_index += 1
    final_file_snippet = "\n".join(file_lines[snippet_start:file_line_index])

    if verbose:
        print(f"Final file snippet: {final_file_snippet}")

    # Ensure the last line of the file is at least 99% similar to the last line of the snippet
    if (
        compute_similarity(
            get_last_non_empty_line("\n".join(snippet_lines)),
            get_last_non_empty_line(final_file_snippet),
        )
        < threshold
    ):
        if verbose:
            print("Last line of snippet is not at least 99% similar to the last line of the file: ")
            print(f"Snippet: {snippet_lines[-1]}")
            print(f"File: {file_lines[file_line_index - 1]}")
        return None

    return final_file_snippet


class SentenceTransformersEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "thenlper/gte-small",
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        print("device", device)
        self._model = SentenceTransformer(model_name).to(device)

        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        batch_embeddings = self._model.encode([text])
        return batch_embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # pool = self._model

        # batch_embeddings = self._model.encode_multi_process(texts, pool)

        # self._model.stop_multi_process_pool(pool)

        return self._model.encode(texts)


def get_details_for_issue(issue_id, organization_slug="sentry"):
    url = f"https://sentry.io/api/0/organizations/{organization_slug}/issues/{issue_id}/"
    headers = {"Authorization": f"Bearer {sentry_auth_token}"}

    response = requests.get(url, headers=headers)
    issue = response.json()

    logger.debug(f"issue: {json.dumps(issue)}")

    if "detail" in issue and issue["detail"] == "The requested resource does not exist":
        raise Exception(f"Could not find issue with id {issue_id}")

    err_msg = issue["title"]

    url = f"https://sentry.io/api/0/organizations/{organization_slug}/issues/{issue_id}/events/latest/"

    response = requests.get(url, headers=headers)
    event = response.json()

    stack_str = ""
    for entry in event["entries"]:
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

    return err_msg, stack_str


def extract_actions(action_response: RESPONSE_TYPE, verbose=False):
    response_str = action_response.response
    thoughts_group = re.search(
        r"<thoughts>(.*?)</thoughts>", response_str, re.MULTILINE | re.DOTALL
    )
    if thoughts_group:
        response_str = response_str.replace(thoughts_group.group(0), "")

    actions_match = re.search(r"<actions>(.*?)</actions>", response_str, re.MULTILINE | re.DOTALL)
    if actions_match:
        actions_str = actions_match.group(1)
    else:
        return None, None

    if verbose:
        print(actions_str)

    try:
        actions = ET.fromstring(f"<actions>{actions_str}</actions>")
    except ET.ParseError:
        print("Error parsing XML")
        actions = None

    return actions, actions_str
