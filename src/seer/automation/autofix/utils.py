import difflib
import random
import re
from functools import lru_cache

from langfuse.decorators import observe

VALID_BRANCH_NAME_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"


@lru_cache(maxsize=1024)
def compute_similarity_cached(text1: str, text2: str, ignore_whitespace=True) -> float:
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


@observe(name="Find original snippet")
def find_original_snippet(
    snippet: str, file_contents: str, threshold=0.8, initial_line_threshold=0.9
) -> tuple[str, int, int] | None:
    """
    This function finds the original snippet of code in a file given a snippet and the file contents.

    Parameters:
    snippet (str): A string containing a snippet of code.
    file_contents (str): A string containing the contents of a file.
    threshold (float): The similarity threshold for the entire snippet.
    initial_line_threshold (float): The similarity threshold for the initial line to start searching.

    The function first searches for a line in the file that matches the first non-empty line of the snippet
    with a similarity above the initial_line_threshold. It then continues from that point to match the
    rest of the snippet, handling ellipsis cases and using the compute_similarity function to compare
    the accumulated snippet with the file contents.

    Returns:
    tuple[str, int, int] | None: A tuple containing the original snippet from the file, start index, and end index,
                                 or None if the snippet could not be found.
    """
    if snippet.strip() == "":
        return None

    snippet_lines = [line for line in snippet.split("\n") if line.strip()]
    file_lines = file_contents.split("\n")

    # Find the first non-empty line in the snippet
    first_snippet_line = next((line for line in snippet_lines if line.strip()), "")

    # Search for a matching initial line in the file
    for start_index, file_line in enumerate(file_lines):
        if compute_similarity_cached(first_snippet_line, file_line) >= initial_line_threshold:
            accumulated_snippet = []
            snippet_index = 0
            file_index = start_index

            while snippet_index < len(snippet_lines) and file_index < len(file_lines):
                file_line = file_lines[file_index].strip()

                if not file_line:
                    file_index += 1
                    continue

                accumulated_snippet.append(file_line)
                similarity = compute_similarity_cached(
                    "\n".join(snippet_lines[: snippet_index + 1]), "\n".join(accumulated_snippet)
                )

                if similarity >= threshold:
                    snippet_index += 1

                file_index += 1

            if snippet_index == len(snippet_lines):
                # All lines in the snippet have been matched
                return "\n".join(file_lines[start_index:file_index]), start_index, file_index

    return None


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
