import difflib
import logging
import os
import random
import re
from typing import List
from xml.etree import ElementTree as ET

from seer.rpc import DummyRpcClient, RpcClient, SentryRpcClient

autofix_logger = logging.getLogger("autofix")

VALID_BRANCH_NAME_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"


def compute_similarity(text1: str, text2: str, ignore_whitespace=True) -> float:
    """
    This function computes the similarity between two pieces of text using the difflib.SequenceMatcher class.

    difflib.SequenceMatcher uses the Ratcliff/Obershelp algorithm: it computes the doubled number of matching characters divided by the total number of characters in the two strings.

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


def find_original_snippet(
    snippet: str, file_contents: str, threshold=0.9
) -> tuple[str, int, int] | None:
    """
    This function finds the original snippet of code in a file given a snippet and the file contents.

    Parameters:
    snippet (str): A string containing a snippet of code.
    file_contents (str): A string containing the contents of a file.

    The function works by splitting the snippet and the file contents into lines and comparing them line by line.
    It uses the compute_similarity function to find the first line of the snippet in the file.
    It then continues comparing the following lines, handling ellipsis cases, until it finds a discrepancy or reaches the end of the snippet.
    If the last line of the snippet is not at least `threshold` similar to the corresponding line in the file, it returns None.
    Otherwise, it returns the original snippet from the file.

    Returns:
    str: The original snippet from the file, or None if the snippet could not be found.
    """
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

    ellipsis_comment_cases = ["// ...", "# ...", "/* ... */"]
    ellipsis_found = False
    snippet_index = 0
    file_line_index = snippet_start
    while snippet_index < len(snippet_lines) and file_line_index < len(file_lines):
        snippet_line = snippet_lines[snippet_index]

        if not ellipsis_found:
            ellipsis_found = snippet_line.strip() == "..." or any(
                s in snippet_line for s in ellipsis_comment_cases
            )
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
            ellipsis_found = False
            if similarity < threshold:
                file_line_index += 1
            else:
                snippet_index += 1
                file_line_index += 1
    final_file_snippet = "\n".join(file_lines[snippet_start:file_line_index])

    # Ensure the last line of the file is at least `threshold` similar to the last line of the snippet
    if (
        compute_similarity(
            get_last_non_empty_line("\n".join(snippet_lines)),
            get_last_non_empty_line(final_file_snippet),
        )
        < threshold
    ):
        return None

    return final_file_snippet, snippet_start, file_line_index


def sanitize_branch_name(title: str) -> str:
    """
    Remove all characters that are not valid in git branch names
    and return a kebab-case branch name from the title.
    """
    kebab_case = title.replace(" ", "-").replace("_", "-").lower()
    sanitized = "".join(c for c in kebab_case if c in VALID_BRANCH_NAME_CHARS)
    return sanitized


def generate_random_string(n=6) -> str:
    """Generate a random n character string."""
    return "".join(random.choice(VALID_BRANCH_NAME_CHARS) for _ in range(n))


def escape_xml_chars(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def escape_xml(s: str, tag: str) -> str:
    match = re.search(rf"<{tag}(\s+[^>]*)?>((.|\n)*?)</{tag}>", s, re.DOTALL)

    if match:
        return s.replace(match.group(2), escape_xml_chars(match.group(2)))

    return s


def escape_multi_xml(s: str, tags: List[str]) -> str:
    for tag in tags:
        s = escape_xml(s, tag)

    return s


def extract_xml_element_text(element: ET.Element, tag: str) -> str | None:
    """
    Extract the text from an XML element with the given tag.

    Args:
        element (ET.Element): The XML element to extract the text from.
        tag (str): The tag of the XML element to extract the text from.

    Returns:
        str: The text of the XML element with the given tag.
    """
    el = element.find(tag)

    if el is not None:
        return (el.text or "").strip()

    return None


def get_sentry_client() -> RpcClient:
    if os.environ.get("NO_SENTRY_INTEGRATION") == "1":
        rpc_client: DummyRpcClient = DummyRpcClient()
        rpc_client.dry_run = True
        return rpc_client
    else:
        return SentryRpcClient()
