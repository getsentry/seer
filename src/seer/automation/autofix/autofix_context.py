from seer.automation.autofix.codebase_context import CodebaseContext
from seer.automation.autofix.models import Stacktrace
from seer.automation.autofix.repo_client import RepoClient


class AutofixContext:
    codebase_context: CodebaseContext | None = None
    repo_client: RepoClient

    def __init__(self, repo_client: RepoClient, base_sha: str):
        self.repo_client = repo_client
        self.base_sha = base_sha

    def load_codebase(self):
        self.codebase_context = CodebaseContext(self.repo_client, self.base_sha)

    def diff_contains_stacktrace_files(self, stacktrace: Stacktrace) -> bool:
        cached_sha = CodebaseContext.get_cached_commit_sha()
        if cached_sha is None:
            raise FileNotFoundError("Cached commit SHA not found")

        changed_files, removed_files = self.repo_client.get_commit_file_diffs(
            cached_sha, self.base_sha
        )

        change_files = set(changed_files + removed_files)
        stacktrace_files = set([frame.filename for frame in stacktrace.frames])

        return bool(change_files.intersection(stacktrace_files))
