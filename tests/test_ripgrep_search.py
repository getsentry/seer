import subprocess
import pytest
from seer.automation.autofix.tools.ripgrep_search import (
    run_ripgrep_in_repo,
    MAX_RIPGREP_LINE_CHARACTER_LENGTH,
    TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH,
)


class DummyCompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_ripgrep_in_repo_success(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd, shell, stdout, stderr, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        # Simulate one match line
        return DummyCompletedProcess(returncode=0, stdout=f"{cwd}/file.txt:1:hello\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_ripgrep_in_repo("/my/repo", ["rg", '"pattern"'])
    # Path should be replaced from repo_dir/ to ./
    assert "./file.txt:1:hello" in result
    # No truncation for short lines
    assert "[LINE TRUNCATED" not in result


def test_run_ripgrep_in_repo_no_results(monkeypatch):
    def fake_run(cmd, cwd, shell, stdout, stderr, text, timeout):
        return DummyCompletedProcess(returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_ripgrep_in_repo("/repo", ["rg", '"q"'])
    assert result == "ripgrep returned: No results found."


def test_run_ripgrep_in_repo_error_exit_code(monkeypatch):
    def fake_run(cmd, cwd, shell, stdout, stderr, text, timeout):
        return DummyCompletedProcess(returncode=2, stderr="bad pattern")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        run_ripgrep_in_repo("/repo", ["rg", '"x"'])
    err = str(excinfo.value)
    assert "ripgrep error (exit 2):" in err
    assert "bad pattern" in err


def test_run_ripgrep_in_repo_timeout(monkeypatch):
    def fake_run(cmd, cwd, shell, stdout, stderr, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        run_ripgrep_in_repo("/repo", ["rg", '"y"'], timeout=0.1)
    assert "ripgrep timed out after 0.1s" in str(excinfo.value)


def test_long_line_truncation(monkeypatch):
    # Create a single line exceeding max line length
    long_line = "A" * (MAX_RIPGREP_LINE_CHARACTER_LENGTH + 10)
    expected_stdout = long_line + "\n"

    def fake_run(cmd, cwd, shell, stdout, stderr, text, timeout):
        # Return our long stdout regardless of PIPE args
        return DummyCompletedProcess(returncode=0, stdout=expected_stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_ripgrep_in_repo("/repo", ["rg", '"z"'])
    assert "...[LINE TRUNCATED TO" in result


def test_total_output_truncation(monkeypatch):
    # Generate output exceeding total results character limit
    line = "file.txt:1:match\n"
    repeat = (TOTAL_RIPGREP_RESULTS_CHARACTER_LENGTH // len(line)) + 10
    expected_stdout = line * repeat

    def fake_run(cmd, cwd, shell, stdout, stderr, text, timeout):
        # Return our large stdout regardless of args
        return DummyCompletedProcess(returncode=0, stdout=expected_stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_ripgrep_in_repo("/repo", ["rg", '"w"'])
    assert "...[RESULT TRUNCATED TO" in result
