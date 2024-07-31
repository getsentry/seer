from langfuse.decorators import observe

from seer.automation.autofix.components.coding.models import FuzzyDiffChunk


@observe(name="Extract diff original/replacement chunks")
def extract_diff_chunks(diff_text: str) -> list[FuzzyDiffChunk]:
    """
    Extract chunks from a diff using the hunk headers (@@ .. @@) as delimiters.

    Args:
    diff_text (str): The full diff text.

    Returns:
    List[DiffChunk]: A list of DiffChunk objects, each containing the hunk header,
                     the original chunk before the diff is applied, and the new chunk after the diff is applied.
    """
    chunks = []
    current_original: list[str] = []
    current_new: list[str] = []
    current_header = ""

    lines = diff_text.split("\n")

    # Skip everything until we hit the first @@ line
    while lines and not lines[0].startswith("@@"):
        lines = lines[1:]

    for line in lines:
        if line.startswith("@@"):
            if current_original or current_new:
                chunks.append(
                    FuzzyDiffChunk(
                        header=current_header,
                        original_chunk="\n".join(current_original),
                        new_chunk="\n".join(current_new),
                    )
                )
                current_original = []
                current_new = []
            current_header = line
        elif line.startswith("-"):
            current_original.append(line[1:])
        elif line.startswith("+"):
            current_new.append(line[1:])
        else:
            current_original.append(line[1:])
            current_new.append(line[1:])

    if current_original or current_new:
        chunks.append(
            FuzzyDiffChunk(
                header=current_header,
                original_chunk="\n".join(current_original),
                new_chunk="\n".join(current_new),
            )
        )

    return chunks
