import uuid

from sentence_transformers import SentenceTransformer

from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixContinuation
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.models import EventDetails, InitializationError, RepoDefinition, Stacktrace
from seer.automation.pipeline import PipelineContext
from seer.automation.state import State
from seer.automation.utils import get_embedding_model
from seer.rpc import RpcClient


class AutofixContext(PipelineContext):
    state: State[AutofixContinuation]
    codebases: dict[int, CodebaseIndex]
    event_manager: AutofixEventManager

    commit_changes: bool = True

    def __init__(
        self,
        sentry_client: RpcClient,
        organization_id: int,
        project_id: int,
        repos: list[RepoDefinition],
        event_manager: AutofixEventManager,
        state: State[AutofixContinuation],
        sha: str | None = None,
        tracking_branch: str | None = None,
        embedding_model: SentenceTransformer | None = None,
    ):
        self.sentry_client = sentry_client
        self.organization_id = organization_id
        self.project_id = project_id
        self.run_id = uuid.uuid4()
        self.codebases = {}

        self.embedding_model = embedding_model or get_embedding_model()

        for repo in repos:
            codebase_index = CodebaseIndex.from_repo_definition(
                organization_id,
                project_id,
                repo,
                sha,
                tracking_branch,
                embedding_model=self.embedding_model,
            )

            if codebase_index:
                self.codebases[codebase_index.repo_info.id] = codebase_index
            else:
                raise InitializationError(f"Failed to load codebase index for repo {repo}")

        self.event_manager = event_manager
        self.state = state

    def has_codebase_index(self, repo: RepoDefinition) -> bool:
        return CodebaseIndex.has_repo_been_indexed(self.organization_id, self.project_id, repo)

    def create_codebase_index(self, repo: RepoDefinition):
        codebase_index = CodebaseIndex.create(
            self.organization_id,
            self.project_id,
            repo,
            embedding_model=self.embedding_model,
        )
        self.codebases[codebase_index.repo_info.id] = codebase_index

    def get_codebase(self, repo_id: int) -> CodebaseIndex:
        codebase = self.codebases[repo_id]

        if codebase is None:
            raise ValueError(f"Codebase with id {repo_id} not found")

        return codebase

    def get_document_and_codebase(
        self, path: str, repo_name: str | None = None, repo_id: int | None = None
    ):
        if repo_name:
            repo_id = next(
                (
                    repo_id
                    for repo_id, codebase in self.codebases.items()
                    if codebase.repo_info.external_slug == repo_name
                ),
                None,
            )
        if repo_id:
            codebase = self.get_codebase(repo_id)
            return codebase, codebase.get_document(path)

        for codebase in self.codebases.values():
            document = codebase.get_document(path)
            if document:
                return codebase, document

        return None, None

    def query_all_codebases(self, query: str, repo_top_k: int = 4):
        chunks = []
        for codebase in self.codebases.values():
            chunks.extend(codebase.query(query, top_k=repo_top_k))

        return chunks

    def diff_contains_stacktrace_files(self, repo_id: int, event_details: EventDetails) -> bool:
        stacktraces = [exception.stacktrace for exception in event_details.exceptions]

        stacktrace_files: set[str] = set()
        for stacktrace in stacktraces:
            for frame in stacktrace.frames:
                stacktrace_files.add(frame.filename)

        codebase = self.get_codebase(repo_id)
        changed_files, removed_files = codebase.repo_client.get_commit_file_diffs(
            codebase.namespace.sha, codebase.repo_client.get_default_branch_head_sha()
        )

        change_files = set(changed_files + removed_files)

        return bool(change_files.intersection(stacktrace_files))

    def _process_stacktrace_paths(self, stacktrace: Stacktrace):
        """
        Annotate a stacktrace with the correct repo each frame is pointing to and fix the filenames
        """
        for codebase in self.codebases.values():
            codebase.process_stacktrace(stacktrace)

    def process_event_paths(self, event: EventDetails):
        """
        Annotate exceptions with the correct repo each frame is pointing to and fix the filenames
        """
        for exception in event.exceptions:
            self._process_stacktrace_paths(exception.stacktrace)

    def cleanup(self):
        for codebase in self.codebases.values():
            codebase.cleanup()
