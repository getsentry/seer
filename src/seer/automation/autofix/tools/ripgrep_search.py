import subprocess

from langfuse.decorators import observe

MAX_RIPGREP_TIMEOUT_SECONDS = 20
MAX_RIPGREP_LINE_CHARACTER_LENGTH = 1024
TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH = 16384


@observe(name="Run ripgrep in repo")
def run_ripgrep_in_repo(
    repo_dir: str, cmd: list[str], timeout: float = MAX_RIPGREP_TIMEOUT_SECONDS
) -> str:
    try:
        result = subprocess.run(
            " ".join(cmd),
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
                f"ripgrep error (exit {result.returncode}):\n{result.stderr.strip()}"
            )

        if result.returncode == 1:
            return "ripgrep returned: No results found."

        # clean out all tmp dirs, we just do a simple replace because these tmp dir strings should be pretty unique.
        output = result.stdout
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

        return output

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ripgrep timed out after {timeout}s")
