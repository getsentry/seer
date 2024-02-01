from seer.automation.autofix.models import Stacktrace
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.repo_client import RepoClient


class AutofixContext:
    codebase: CodebaseIndex

    def __init__(self, organization_id: int, project_id: int, repo_client: RepoClient):
        self.codebase = CodebaseIndex.from_repo_client(organization_id, project_id, repo_client)

    def diff_contains_stacktrace_files(self, stacktrace: Stacktrace) -> bool:
        if self.codebase.repo_info is None:
            raise FileNotFoundError("Cached commit SHA not found")

        changed_files, removed_files = self.codebase.repo_client.get_commit_file_diffs(
            self.codebase.repo_info.sha, self.codebase.repo_client.get_default_branch_head_sha()
        )

        change_files = set(changed_files + removed_files)
        stacktrace_files = set([frame.filename for frame in stacktrace.frames])

        return bool(change_files.intersection(stacktrace_files))
