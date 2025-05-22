import logging
import os
import subprocess
from pathlib import Path

import sentry_sdk
from langfuse.decorators import observe

MAX_RIPGREP_TIMEOUT_SECONDS = 20
MAX_RIPGREP_LINE_CHARACTER_LENGTH = 1024
TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH = 16384

logger = logging.getLogger(__name__)


def calculate_dynamic_timeout(repo_dir: str, base_timeout: float = MAX_RIPGREP_TIMEOUT_SECONDS) -> float:
    """Calculate appropriate timeout based on repository size."""
    try:
        # Get total size of the repository
        total_size = 0
        for dirpath, _, filenames in os.walk(repo_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
        
        # Convert to MB for easier calculation
        size_mb = total_size / (1024 * 1024)
        
        # Scale timeout linearly with size, with some constraints
        # Base size is considered 100MB
        if size_mb <= 100:
            return base_timeout
        elif size_mb <= 500:
            # Scale up to 2x for medium repos
            scale_factor = 1.0 + (size_mb - 100) / 400
            return min(base_timeout * scale_factor, base_timeout * 2)
        else:
            # For very large repos, cap at 3x
            return min(base_timeout * 3, 60)  # Max 60 seconds
    except Exception as e:
        logger.exception(f"Error calculating dynamic timeout: {e}")
        return base_timeout  # Fall back to default timeout


@observe(name="Run ripgrep in repo")
@sentry_sdk.trace
def run_ripgrep_in_repo(
    repo_dir: str, cmd: list[str], timeout: float = None
) -> str:
    if timeout is None:
        timeout = calculate_dynamic_timeout(repo_dir)
    try:
        prepared_cmd = " ".join(cmd)
        result = subprocess.run(
            prepared_cmd,
            cwd=repo_dir,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        # ripgrep returns 1 when it finds *nothing* (not an error),
        # >1 for real errors (bad pattern, etc.)
        if result.returncode not in (0, 1):
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            if "regex parse error" in stderr and "--fixed-strings" not in cmd:
                # try again with fixed strings on a regex parse error
                cmd.insert(1, "--fixed-strings")
                return run_ripgrep_in_repo(repo_dir, cmd, timeout)
            raise RuntimeError(
                f"Ran ripgrep with command: `{prepared_cmd}`\n\nripgrep Error (exit {result.returncode}):\n{stderr}"
            )

        output = result.stdout.decode("utf-8", errors="replace")

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
        return (
            f"Ran ripgrep with command: `{prepared_cmd}`\n\n"
            f"Search timed out after {timeout}s. The repository may be too large or the search pattern too general.\n"
            f"Consider:\n"
            f"- Using a more specific search pattern\n"
            f"- Adding include patterns (e.g., '*.py' for Python files only)\n"
            f"- Adding exclude patterns (e.g., exclude large directories like 'node_modules')\n"
            f"- Breaking your search into multiple smaller searches"
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
