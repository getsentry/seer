import logging

from langfuse.decorators import observe

from seer.automation.autofix.components.coding.models import FuzzyDiffChunk, PlanTaskPromptXml
from seer.automation.autofix.utils import find_original_snippet
from seer.automation.models import FileChange
from seer.langfuse import append_langfuse_trace_tags

logger = logging.getLogger(__name__)


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


@observe(name="Convert task to file create")
def task_to_file_create(task: PlanTaskPromptXml) -> FileChange:
    """
    Convert a PlanTaskPromptXml to a FileChange object for file creation.

    Args:
    task (PlanTaskPromptXml): The task to convert.

    Returns:
    FileChange: A FileChange object representing the file creation task.
    """
    if task.type != "file_create":
        raise ValueError(f"Expected file_create task, got: {task.type}")

    diff_chunks = extract_diff_chunks(task.diff)
    if len(diff_chunks) != 1:
        raise ValueError(
            f"Expected exactly one diff chunk for file creation, got {len(diff_chunks)}"
        )

    return FileChange(
        change_type="create",
        path=task.file_path,
        new_snippet=diff_chunks[0].new_chunk,
    )


@observe(name="Convert task to file delete")
def task_to_file_delete(task: PlanTaskPromptXml) -> FileChange:
    """
    Convert a PlanTaskPromptXml to a FileChange object for file deletion.

    Args:
    task (PlanTaskPromptXml): The task to convert.

    Returns:
    FileChange: A FileChange object representing the file deletion task.
    """
    if task.type != "file_delete":
        raise ValueError(f"Expected file_delete task, got: {task.type}")

    return FileChange(
        change_type="delete",
        path=task.file_path,
    )


@observe(name="Convert task to file change")
def task_to_file_change(task: PlanTaskPromptXml, file_content: str) -> list[FileChange]:
    """
    Convert a PlanTaskPromptXml to a list of FileChange objects for file changes.

    Args:
    task (PlanTaskPromptXml): The task to convert.
    file_content (str): The content of the file being changed.

    Returns:
    list[FileChange]: A list of FileChange objects representing the file change tasks.
    """
    if task.type != "file_change":
        raise ValueError(f"Expected file_change task, got: {task.type}")

    changes = []
    diff_chunks = extract_diff_chunks(task.diff)

    for chunk in diff_chunks:
        result = find_original_snippet(
            chunk.original_chunk,
            file_content,
            threshold=0.75,
            initial_line_threshold=0.95,
        )

        if result:
            original_snippet = result[0]
            changes.append(
                FileChange(
                    change_type="edit",
                    path=task.file_path,
                    reference_snippet=original_snippet,
                    new_snippet=chunk.new_chunk,
                    description=task.description,
                )
            )
        else:
            logger.info(f"Original snippet not found in file {task.file_path}")
            append_langfuse_trace_tags(["skipped_file_change:snippet_not_found"])

    return changes
