import random

from langfuse.decorators import observe

# Define a separate character set for random string generation that excludes slashes and dashes
VALID_RANDOM_SUFFIX_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
from rapidfuzz import fuzz, process

VALID_BRANCH_NAME_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-/"


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
    snippet: str, file_contents: str, threshold: float = 0.8, initial_line_threshold: float = 0.9
) -> tuple[str, int, int] | None:
    """
    This function finds the original snippet of code in a file given a snippet and the file contents.
    Uses rapidfuzz for fast and accurate fuzzy string matching.

    Parameters:
    snippet (str): A string containing a snippet of code.
    file_contents (str): A string containing the contents of a file.
    threshold (float): The similarity threshold for the entire snippet.
    initial_line_threshold (float): The similarity threshold for the initial line to start searching.

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

    best_match = None
    best_score = -1.0

    # Find all potential matches
    matches = process.extract(
        first_snippet_line,
        file_lines,
        scorer=fuzz.ratio,  # Use exact ratio for first line
        score_cutoff=initial_line_threshold * 100,
        limit=None,
    )

    for file_line, score, start_index in matches:
        # For each potential starting point, try to match the full snippet
        candidate_end = min(start_index + len(snippet_lines) + 5, len(file_lines))
        candidate_snippet = "\n".join(file_lines[start_index:candidate_end])

        # Compare the full snippets using token_set_ratio for better matching
        full_score = fuzz.token_set_ratio(snippet, candidate_snippet) / 100.0

        if full_score > best_score and full_score >= threshold:
            # Find the exact end index by matching the content
            actual_end = start_index
            snippet_index = 0

            while actual_end < len(file_lines) and snippet_index < len(snippet_lines):
                # Skip empty lines in the file but don't increment snippet_index
                if not file_lines[actual_end].strip():
                    actual_end += 1
                    continue

                # Skip empty lines in the snippet but don't increment actual_end
                if not snippet_lines[snippet_index].strip():
                    snippet_index += 1
                    continue

                # Compare current lines
                line_score = (
                    fuzz.ratio(snippet_lines[snippet_index], file_lines[actual_end]) / 100.0
                )
                if line_score >= threshold:
                    actual_end += 1
                    snippet_index += 1
                else:
                    # If this line doesn't match well, try the next file line
                    actual_end += 1

            # Make sure we've matched all snippet lines
            if snippet_index >= len(snippet_lines):
                best_score = full_score
                best_match = (
                    "\n".join(file_lines[start_index:actual_end]),
                    start_index,
                    actual_end,
                )

    return best_match


def sanitize_branch_name(title: str) -> str:
    """
    Remove all characters that are not valid in git branch names
    and return a kebab-case branch name from the title.
    """
    kebab_case = title.replace(" ", "-").replace("_", "-").lower()
    sanitized = "".join(c for c in kebab_case if c in VALID_BRANCH_NAME_CHARS)
    sanitized = sanitized.rstrip("/")
    return sanitized


def generate_random_string(n=6) -> str:
    """Generate a random n character string."""
    return "".join(random.choice(VALID_RANDOM_SUFFIX_CHARS) for _ in range(n))


def remove_code_backticks(text: str) -> str:
    """Remove code backticks from a string."""
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
