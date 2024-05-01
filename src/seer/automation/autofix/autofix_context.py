import textwrap
from typing import cast

import sentry_sdk
from sentence_transformers import SentenceTransformer

from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    ChangesStep,
    CodebaseState,
    CommittedPullRequestDetails,
)
from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import QueryResultDocumentChunk
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.state import CodebaseStateManager
from seer.automation.models import EventDetails, FileChange, RepoDefinition, Stacktrace
from seer.automation.pipeline import PipelineContext
from seer.automation.state import State
from seer.automation.utils import get_embedding_model
from seer.rpc import RpcClient


class AutofixCodebaseStateManager(CodebaseStateManager):
    state: State[AutofixContinuation]

    def store_file_change(self, file_change: FileChange):
        with self.state.update() as state:
            codebase_state = state.codebases[self.repo_id]
            codebase_state.file_changes.append(file_change)

            autofix_logger.info(f"Stored file change for repo {self.repo_id}, new state: {state}")

    def get_file_changes(self) -> list[FileChange]:
        return self.state.get().codebases[self.repo_id].file_changes


class AutofixContext(PipelineContext):
    state: State[AutofixContinuation]
    codebases: dict[int, CodebaseIndex]
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

        if not skip_loading_codebase:
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
                    self.codebases[codebase_index.repo_info.id] = codebase_index
                    with state.update() as cur:
                        if codebase_index.repo_info.id not in cur.codebases:
                            cur.codebases[codebase_index.repo_info.id] = CodebaseState(
                                repo_id=codebase_index.repo_info.id,
                                namespace_id=codebase_index.namespace.id,
                                file_changes=[],
                            )

        self.event_manager = event_manager
        self.state = state

    def has_codebase_index(self, repo: RepoDefinition) -> bool:
        for codebase in self.codebases.values():
            if codebase.repo_info.external_id == repo.external_id:
                return True

        return False

    def has_missing_codebase_indexes(self) -> bool:
        for repo in self.repos:
            if not self.has_codebase_index(repo):
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

    def query_all_codebases(self, query: str, top_k: int = 4) -> list[QueryResultDocumentChunk]:
        """
        Queries all codebases for top_k chunks matching the specified query and returns the only the overall top_k closest matches.
        """
        chunks: list[QueryResultDocumentChunk] = []
        for codebase in self.codebases.values():
            chunks.extend(codebase.query(query, top_k=2 * top_k))

        chunks.sort(key=lambda x: x.distance)

        return chunks[:top_k]

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

    def commit_changes(self, repo_id: int | None = None):
        with self.state.update() as state:
            for codebase_state in state.codebases.values():
                if repo_id == None or codebase_state.repo_id == repo_id:
                    changes_step = state.find_step(id="changes")
                    if not changes_step:
                        raise ValueError("Changes step not found")
                    changes_step = cast(ChangesStep, changes_step)
                    change_state = next(
                        (
                            change
                            for change in changes_step.changes
                            if change.repo_id == codebase_state.repo_id
                        ),
                        None,
                    )
                    if codebase_state.file_changes and change_state:
                        repo_info = CodebaseIndex.get_repo_info_from_db(codebase_state.repo_id)
                        if repo_info is None:
                            raise ValueError(
                                f"Repo info not found for repo id {codebase_state.repo_id}"
                            )

                        repo_client = RepoClient.from_repo_info(repo_info)
                        branch_ref = repo_client.create_branch_from_changes(
                            pr_title=change_state.title,
                            file_changes=codebase_state.file_changes,
                        )

                        if branch_ref is None:
                            autofix_logger.warning(f"Failed to create branch from changes")
                            return None

                        pr_title = f"""🤖 {change_state.title}"""

                        ref_note = ""
                        org_slug = self._get_org_slug(state.request.organization_id)
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
                            👋 Hi there! This PR was automatically generated 🤖
                            {user_line}

                            {ref_note}
                            {description}

                            ### 📣 Instructions for the reviewer which is you, yes **you**:
                            - **If these changes were incorrect, please close this PR and comment explaining why.**
                            - **If these changes were incomplete, please continue working on this PR then merge it.**
                            - **If you are feeling confident in my changes, please merge this PR.**

                            This will greatly help us improve the autofix system. Thank you! 🙏

                            If there are any questions, please reach out to the [AI/ML Team](https://github.com/orgs/getsentry/teams/machine-learning-ai) on [#proj-autofix](https://sentry.slack.com/archives/C06904P7Z6E)

                            ### 🤓 Stats for the nerds:
                            Prompt tokens: **{prompt_tokens}**
                            Completion tokens: **{completion_tokens}**
                            Total tokens: **{total_tokens}**"""
                        ).format(
                            user_line=(
                                f"\nTriggered by {state.request.invoking_user.display_name}"
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
                            pr_number=pr.number, pr_url=pr.html_url
                        )

    def _get_org_slug(self, organization_id: int) -> str | None:
        slug: str | None = None
        try:
            response = self.sentry_client.call("get_organization_slug", org_id=organization_id)
            slug = None if response is None else response.get("slug", None)
            if slug == None:
                autofix_logger.warn(
                    f"Slug lookup call for organization {organization_id} succeeded but returned value None."
                )
        except Exception as e:
            autofix_logger.warn(f"Failed to get slug for organization {organization_id}")
            autofix_logger.exception(e)
            sentry_sdk.capture_exception(e)
            slug = None
        return slug

    def cleanup(self):
        for codebase in self.codebases.values():
            codebase.cleanup()
