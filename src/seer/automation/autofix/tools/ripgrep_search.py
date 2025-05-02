import logging
import subprocess
from pathlib import Path

from langfuse.decorators import observe

MAX_RIPGREP_TIMEOUT_SECONDS = 20
MAX_RIPGREP_LINE_CHARACTER_LENGTH = 1024
TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH = 16384

logger = logging.getLogger(__name__)


@observe(name="Run ripgrep in repo")
def run_ripgrep_in_repo(
    repo_dir: str, cmd: list[str], timeout: float = MAX_RIPGREP_TIMEOUT_SECONDS
) -> str:
    try:
        prepared_cmd = " ".join(cmd)
        result = subprocess.run(
            prepared_cmd,
            cwd=repo_dir,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        # ripgrep returns 1 when it finds *nothing* (not an error),
        # >1 for real errors (bad pattern, etc.)
        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"Ran ripgrep with command: `{prepared_cmd}`\n\nripgrep Error (exit {result.returncode}):\n{result.stderr.strip()}"
            )

        output = result.stdout

        if result.returncode == 1 or not output:
            try:
                list_directory_contents(
                    repo_dir
                )  # Debug log the repo directory contents and filetypes to langfuse
            except Exception as e:
                logger.exception(e)

            return (
                f"Ran ripgrep with command: `{prepared_cmd}`\n\nripgrep returned: No results found."
            )

        # clean out all tmp dirs, we just do a simple replace because these tmp dir strings should be pretty unique.

        output = output.replace(repo_dir + "/", "./")

        lines = []
        for line in output.split("\n"):
            if len(line) > MAX_RIPGREP_LINE_CHARACTER_LENGTH:
                lines.append(
                    f"{line[:MAX_RIPGREP_LINE_CHARACTER_LENGTH]}...[LINE TRUNCATED TO {MAX_RIPGREP_LINE_CHARACTER_LENGTH} CHARACTERS]"
                )
            else:
                lines.append(line)

        output = "\n".join(lines)

        if len(output) > TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH:
            output = f"{output[:TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH]}...[RESULT TRUNCATED TO {TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH} CHARACTERS, RUN A MORE SPECIFIC QUERY TO SEE THE REMAINDER OF THE RESULT]"

        return f"Ran ripgrep with command: `{prepared_cmd}`\n\nResult:\n{output}"

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Ran ripgrep with command: `{prepared_cmd}`\n\nripgrep timed out after {timeout}s"
        )


@observe(name="DEBUG List directory contents")
def list_directory_contents(directory="."):
    # Get all entries in the directory
    output = ""
    for entry in Path(directory).iterdir():
        # Get file type
        if entry.is_symlink():
            file_type = "symlink"
        elif entry.is_dir():
            file_type = "directory"
        elif entry.is_file():
            file_type = "file"
        else:
            file_type = "unknown"

        # Get file name
        name = entry.name

        output += f"{file_type:<10} {name}\n"

    return output
