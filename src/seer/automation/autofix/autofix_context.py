import textwrap
from typing import Mapping, cast

import sentry_sdk
from sentence_transformers import SentenceTransformer

from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    ChangesStep,
    CodebaseState,
    CommittedPullRequestDetails,
)
from seer.automation.autofix.state import ContinuationState
from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.file_patches import make_file_patches
from seer.automation.codebase.models import BaseDocument, QueryResultDocumentChunk
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.state import CodebaseStateManager
from seer.automation.codebase.utils import potential_frame_match
from seer.automation.models import EventDetails, FileChange, FilePatch, RepoDefinition, Stacktrace
from seer.automation.pipeline import PipelineContext
from seer.automation.state import State
from seer.automation.utils import get_embedding_model, get_sentry_client
from seer.db import DbPrIdToAutofixRunIdMapping, Session
from seer.rpc import RpcClient

RepoExternalId = str
RepoInternalId = int
RepoKey = RepoExternalId | RepoInternalId
RepoIdentifiers = tuple[RepoExternalId, RepoInternalId]


class AutofixCodebaseStateManager(CodebaseStateManager):
    state: State[AutofixContinuation]

    def store_file_change(self, file_change: FileChange):
        with self.state.update() as state:
            codebase_state = state.codebases[self.repo_external_id]
            codebase_state.file_changes.append(file_change)

    def get_file_changes(self) -> list[FileChange]:
        return self.state.get().codebases[self.repo_external_id].file_changes


class AutofixContext(PipelineContext):
    state: State[AutofixContinuation]
    codebases: dict[str, CodebaseIndex]
    repos: list[RepoDefinition]

    event_manager: AutofixEventManager
    sentry_client: RpcClient

    def __init__(
        self,
        state: State[AutofixContinuation],
        sentry_client: RpcClient,
        event_manager: AutofixEventManager,
        embedding_model: SentenceTransformer | None = None,
        skip_loading_codebase: bool = False,
    ):
        request = state.get().request

        self.organization_id = request.organization_id
        self.project_id = request.project_id
        self.repos = request.repos

        self.codebases = {}

        self.sentry_client = sentry_client
        self.embedding_model = embedding_model or get_embedding_model()

        no_codebase_indexes = skip_loading_codebase or request.options.disable_codebase_indexing

        if no_codebase_indexes:
            with state.update() as cur:
                for repo in request.repos:
                    if repo.external_id not in cur.codebases:
                        cur.codebases[repo.external_id] = CodebaseState(
                            file_changes=[],
                            repo_external_id=repo.external_id,
                        )
        else:
            for repo in request.repos:
                codebase_index = CodebaseIndex.from_repo_definition(
                    request.organization_id,
                    request.project_id,
                    repo,
                    request.base_commit_sha,
                    None,
                    state=state,
                    state_manager_class=AutofixCodebaseStateManager,
                    embedding_model=self.embedding_model,
                )

                if codebase_index:
                    self.codebases[codebase_index.repo_info.external_id] = codebase_index
                    with state.update() as cur:
                        if codebase_index.repo_info.external_id not in cur.codebases:
                            cur.codebases[codebase_index.repo_info.external_id] = CodebaseState(
                                repo_id=codebase_index.repo_info.id,
                                namespace_id=codebase_index.namespace.id,
                                file_changes=[],
                                repo_external_id=codebase_index.repo_info.external_id,
                            )

        self.event_manager = event_manager
        self.state = state
        self.skip_loading_codebase = no_codebase_indexes

        autofix_logger.info(
            f"AutofixContext initialized with run_id {self.run_id}, {'without codebase indexing' if self.skip_loading_codebase else 'with codebase indexing'}"
        )

    @classmethod
    def from_run_id(cls, run_id: int):
        state = ContinuationState.from_id(run_id, model=AutofixContinuation)
        with state.update() as cur:
            cur.mark_triggered()

        event_manager = AutofixEventManager(state)

        return cls(state, get_sentry_client(), event_manager)

    @property
    def run_id(self) -> int:
        return self.state.get().run_id

    @property
    def signals(self) -> list[str]:
        return self.state.get().signals

    @signals.setter
    def signals(self, value: list[str]):
        with self.state.update() as state:
            state.signals = value

    def repos_by_key(self) -> Mapping[RepoKey, RepoDefinition]:
        repos_by_key: dict[RepoKey, RepoDefinition] = {
            repo.external_id: repo for repo in self.repos
        }
        for codebase_state in self.codebases.values():
            external_id = codebase_state.repo_info.external_id

            repo = next(
                (repo for repo in self.repos if repo.external_id == external_id),
                None,
            )
            if repo:
                repos_by_key[codebase_state.repo_info.id] = repo

        return repos_by_key

    def has_missing_codebase_indexes(self) -> bool:
        for repo in self.repos:
            codebase = self.codebases.get(repo.external_id)
            if codebase is None or not codebase.workspace.is_ready():
                return True

        return False

    def has_codebase_indexing_run(self) -> bool:
        return self.state.get().find_step(id=self.event_manager.indexing_step.id) is not None

    def create_codebase_index(self, repo: RepoDefinition) -> CodebaseIndex:
        namespace_id = CodebaseIndex.create(
            self.organization_id,
            self.project_id,
            repo,
        )
        codebase_index = CodebaseIndex.index(
            namespace_id,
            embedding_model=self.embedding_model,
        )

        self.codebases[codebase_index.repo_info.id] = codebase_index

        return codebase_index

    def get_repo_client(self, repo_name: str | None = None, repo_external_id: str | None = None):
        """
        Gets a repo client for the current single repo or for a given repo name.
        If there are more than 1 repos, a repo name must be provided.
        """
        repo_client: RepoClient | None = None
        if len(self.repos) == 1:
            repo_client = RepoClient.from_repo_definition(self.repos[0], "read")
        elif repo_name:
            repo = next((r for r in self.repos if r.full_name == repo_name), None)

            if not repo:
                raise ValueError(f"Repo {repo_name} not found.")

            repo_client = RepoClient.from_repo_definition(repo, "read")
        elif repo_external_id:
            repo = next((r for r in self.repos if r.external_id == repo_external_id), None)

            if not repo:
                raise ValueError(f"Repo {repo_external_id} not found.")

            repo_client = RepoClient.from_repo_definition(repo, "read")
        else:
            raise ValueError("Please provide a repo name because you have multiple repos.")

        return repo_client

    def get_file_contents(
        self, path: str, repo_name: str | None = None, ignore_local_changes: bool = False
    ) -> str | None:
        if repo_name is None:
            raise ValueError("Please provide a repo name because you have multiple repos.")
        repo_client = self.get_repo_client(repo_name)

        file_contents = repo_client.get_file_content(path)

        if not ignore_local_changes:
            cur_state = self.state.get()
            repo_file_changes = cur_state.codebases[repo_client.repo_external_id].file_changes
            current_file_changes = list(filter(lambda x: x.path == path, repo_file_changes))
            for file_change in current_file_changes:
                file_contents = file_change.apply(file_contents)

        return file_contents

    def query_all_codebases(self, query: str, top_k: int = 4) -> list[QueryResultDocumentChunk]:
        """
        Queries all codebases for top_k chunks matching the specified query and returns the only the overall top_k closest matches.
        """
        chunks: list[QueryResultDocumentChunk] = []
        for codebase in self.codebases.values():
            chunks.extend(codebase.query(query, top_k=2 * top_k))

        chunks.sort(key=lambda x: x.distance)

        return chunks[:top_k]

    def _process_stacktrace_paths(self, stacktrace: Stacktrace):
        """
        Annotate a stacktrace with the correct repo each frame is pointing to and fix the filenames
        """
        for repo in self.repos:
            repo_client = RepoClient.from_repo_definition(repo, "read")

            valid_file_paths = repo_client.get_valid_file_paths()
            for frame in stacktrace.frames:
                if frame.in_app and frame.repo_id is None:
                    if frame.filename in valid_file_paths:
                        frame.repo_name = repo.full_name
                    else:
                        for valid_path in valid_file_paths:
                            if potential_frame_match(valid_path, frame):
                                frame.repo_name = repo.full_name
                                frame.filename = valid_path
                                break

    def process_event_paths(self, event: EventDetails):
        """
        Annotate exceptions with the correct repo each frame is pointing to and fix the filenames
        """
        for exception in event.exceptions:
            self._process_stacktrace_paths(exception.stacktrace)
        for thread in event.threads:
            if thread.stacktrace:
                self._process_stacktrace_paths(thread.stacktrace)

    def commit_changes(self, repo_external_id: str | None = None, repo_id: int | None = None):
        with self.state.update() as state:
            for codebase_state in state.codebases.values():
                if (
                    (repo_external_id is None and repo_id is None)
                    or codebase_state.repo_external_id == repo_external_id
                    # TODO: Remove this when repo_id is removed from the model
                    or codebase_state.repo_id == repo_id
                ):
                    changes_step = state.find_step(id="changes")
                    if not changes_step:
                        raise ValueError("Changes step not found")
                    changes_step = cast(ChangesStep, changes_step)
                    change_state = next(
                        (
                            change
                            for change in changes_step.changes
                            if change.repo_external_id == codebase_state.repo_external_id
                            # TODO: Remove this when repo_id is removed from the model
                            or change.repo_id == codebase_state.repo_id
                        ),
                        None,
                    )
                    if codebase_state.file_changes and change_state:
                        key = codebase_state.repo_external_id or codebase_state.repo_id

                        if key is None:
                            raise ValueError("Repo key not found")

                        repo_definition = self.repos_by_key().get(key)

                        if repo_definition is None:
                            raise ValueError(f"Repo definition not found for key {key}")

                        repo_client = RepoClient.from_repo_definition(repo_definition, "write")

                        branch_ref = repo_client.create_branch_from_changes(
                            pr_title=change_state.title,
                            file_changes=codebase_state.file_changes,
                        )

                        if branch_ref is None:
                            autofix_logger.warning("Failed to create branch from changes")
                            return None

                        pr_title = f"""🤖 {change_state.title}"""

                        ref_note = ""
                        org_slug = self.get_org_slug(state.request.organization_id)
                        if org_slug:
                            issue_url = f"https://sentry.io/organizations/{org_slug}/issues/{state.request.issue.id}/"
                            issue_link = (
                                f"[{state.request.issue.short_id}]({issue_url})"
                                if state.request.issue.short_id
                                else issue_url
                            )
                            ref_note = f"Fixes {issue_link}\n"

                        pr_description = textwrap.dedent(
                            """\
                            👋 Hi there! This PR was automatically generated by Autofix 🤖
                            {user_line}

                            {ref_note}
                            {description}

                            If you have any questions or feedback for the Sentry team about this fix, please email [autofix@sentry.io](mailto:autofix@sentry.io) with the Run ID (see below).

                            ### 🤓 Stats for the nerds:
                            Run ID: **{run_id}**
                            Prompt tokens: **{prompt_tokens}**
                            Completion tokens: **{completion_tokens}**
                            Total tokens: **{total_tokens}**"""
                        ).format(
                            run_id=state.run_id,
                            user_line=(
                                f"\nThis fix was triggered by {state.request.invoking_user.display_name}"
                                if state.request.invoking_user
                                else ""
                            ),
                            description=change_state.description,
                            ref_note=ref_note,
                            prompt_tokens=state.usage.prompt_tokens,
                            completion_tokens=state.usage.completion_tokens,
                            total_tokens=state.usage.total_tokens,
                        )

                        pr = repo_client.create_pr_from_branch(branch_ref, pr_title, pr_description)

                        change_state.pull_request = CommittedPullRequestDetails(
                            pr_number=pr.number, pr_url=pr.html_url, pr_id=pr.id
                        )

                        with Session() as session:
                            pr_id_mapping = DbPrIdToAutofixRunIdMapping(
                                provider=repo_client.provider,
                                pr_id=pr.id,
                                run_id=state.run_id,
                            )
                            session.add(pr_id_mapping)
                            session.commit()

    def get_org_slug(self, organization_id: int) -> str | None:
        slug: str | None = None
        try:
            response = self.sentry_client.call("get_organization_slug", org_id=organization_id)
            slug = None if response is None else response.get("slug", None)
            if slug is None:
                autofix_logger.warn(
                    f"Slug lookup call for organization {organization_id} succeeded but returned value None."
                )
        except Exception as e:
            autofix_logger.warn(f"Failed to get slug for organization {organization_id}")
            autofix_logger.exception(e)
            sentry_sdk.capture_exception(e)
            slug = None
        return slug

    def make_file_patches(
        self, file_changes: list[FileChange], repo_name: str
    ) -> tuple[list[FilePatch], str]:
        document_paths = list(set([file_change.path for file_change in file_changes]))

        if not document_paths:
            return [], ""

        original_documents = [
            BaseDocument(
                path=path,
                text=self.get_file_contents(path, repo_name=repo_name, ignore_local_changes=True)
                or "",
            )
            for path in document_paths
        ]

        return make_file_patches(file_changes, document_paths, original_documents)

    def cleanup(self):
        for codebase in self.codebases.values():
            codebase.cleanup()
