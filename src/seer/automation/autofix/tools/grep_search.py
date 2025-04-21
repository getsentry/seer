import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from seer.dependency_injection import copy_modules_initializer

logger = logging.getLogger(__name__)

GREP_TIMEOUT_SECONDS = 45
MAX_GREP_LINE_CHARACTER_LENGTH = 1024
TOTAL_GREP_RESULTS_CHARACTER_LENGTH = 16384


def _run_grep_in_repo(
    repo_name: str,
    cmd_args: list[str],
    tmp_repo_dir: str,
) -> str | None:
    """Runs the grep command in a specific repository directory."""
    try:
        # Run the grep command in the repo directory
        try:
            process = subprocess.run(
                cmd_args,
                shell=False,
                cwd=tmp_repo_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=GREP_TIMEOUT_SECONDS,
            )

            # Check if error is due to "is a directory" and retry with -r flag
            if (
                process.returncode != 0
                and process.returncode != 1
                and "is a directory" in process.stderr.lower()
            ):
                if "-r" not in cmd_args and "--recursive" not in cmd_args:
                    recursive_cmd_args = cmd_args.copy()
                    # Insert -r after the command itself (e.g., grep -r ...)
                    # Handle potential edge cases like `rg -r` already existing
                    if cmd_args[0] == "grep":
                        recursive_cmd_args.insert(1, "-r")
                    else:  # Assume ripgrep or similar, add --recursive
                        recursive_cmd_args.insert(1, "--recursive")

                    process = subprocess.run(
                        recursive_cmd_args,
                        shell=False,
                        cwd=tmp_repo_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=GREP_TIMEOUT_SECONDS,
                    )

            if (
                process.returncode != 0 and process.returncode != 1
            ):  # grep returns 1 when no matches found
                return f"Results from {repo_name}: {process.stderr}"
            elif process.stdout:
                final_output = process.stdout
                # Each line is a grep result, -A, -B, -C are ways to get lines before, after, and around the match
                if "-A" not in cmd_args and "-B" not in cmd_args and "-C" not in cmd_args:
                    lines = process.stdout.split("\n")
                    final_output = ""
                    for line in lines:
                        if len(line) > MAX_GREP_LINE_CHARACTER_LENGTH:
                            line = (
                                line[:MAX_GREP_LINE_CHARACTER_LENGTH]
                                + "...[TRUNCATED: line too long to display]"
                            )
                        final_output += line + "\n"

                if len(final_output) > TOTAL_GREP_RESULTS_CHARACTER_LENGTH:
                    final_output = (
                        final_output[:TOTAL_GREP_RESULTS_CHARACTER_LENGTH]
                        + "...[GREP RESULTS TRUNCATED: too long to display, try narrowing your search]"
                    )

                return f"Results from {repo_name}:\n------\n{final_output}\n------"
            else:
                return f"Results from {repo_name}: no results found."
        except subprocess.TimeoutExpired:
            return f"Results from {repo_name}: command timed out. Try narrowing your search."
    except Exception as e:
        logger.exception(f"Error running grep in repo {repo_name}: {e}")
        return f"Error in repo {repo_name}: {str(e)}"


def run_grep_search(
    cmd_args: list[str], repo_names: list[str], tmp_dirs: dict[str, tuple[str, str]]
) -> str:
    """
    Performs the grep search across one or more repositories.

    Args:
        cmd_args: The parsed grep command arguments.
        repo_names: List of repository names to search within.
        tmp_dirs: Dictionary mapping repo names to their temporary directory paths.

    Returns:
        A string containing the combined results from all repositories.
    """
    all_results = []

    def run_grep_for_repo(repo_name: str):
        if repo_name not in tmp_dirs:
            return None
        _, tmp_repo_dir = tmp_dirs[repo_name]
        if not tmp_repo_dir:
            return None
        return _run_grep_in_repo(repo_name, cmd_args, tmp_repo_dir)

    if len(repo_names) == 1:
        result = run_grep_for_repo(repo_names[0])
        if result:
            all_results.append(result)
    else:
        with ThreadPoolExecutor(initializer=copy_modules_initializer()) as executor:
            future_to_repo = {
                executor.submit(run_grep_for_repo, repo_name): repo_name for repo_name in repo_names
            }
            for future in as_completed(future_to_repo):
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    repo_name = future_to_repo[future]
                    logger.exception(f"Error processing grep result for repo {repo_name}: {e}")
                    all_results.append(f"Error processing result for {repo_name}: {str(e)}")

    if not all_results:
        return "No results found."

    return "\n\n".join(all_results)
