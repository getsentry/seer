import functools
import logging
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Mapping, Set, Tuple

import sentry_sdk
from github import UnknownObjectException
from pydantic import ValidationError

from seer.automation.agent.models import Message
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRunMemory,
    CodebaseChange,
    CommittedPullRequestDetails,
)
from seer.automation.autofix.state import ContinuationState
from seer.automation.codebase.file_patches import make_file_patches
from seer.automation.codebase.models import BaseDocument
from seer.automation.codebase.repo_client import (
    RepoClient,
    RepoClientType,
    autocorrect_repo_name,
    get_repo_client,
)
from seer.automation.codebase.utils import potential_frame_match
from seer.automation.models import EventDetails, FileChange, FilePatch, RepoDefinition, Stacktrace
from seer.automation.pipeline import PipelineContext
from seer.automation.state import State
from seer.automation.summarize.issue import IssueSummaryWithScores
from seer.automation.utils import AgentError
from seer.db import DbIssueSummary, DbPrIdToAutofixRunIdMapping, DbRunMemory, Session
from seer.dependency_injection import copy_modules_initializer, inject, injected
from seer.rpc import RpcClient

logger = logging.getLogger(__name__)

RepoExternalId = str
RepoInternalId = int
RepoKey = RepoExternalId | RepoInternalId
RepoIdentifiers = tuple[RepoExternalId, RepoInternalId]


class AutofixContext(PipelineContext):
    state: State[AutofixContinuation]
    repos: list[RepoDefinition]

    event_manager: AutofixEventManager
    sentry_client: RpcClient

    @inject
    def __init__(
        self,
        state: State[AutofixContinuation],
        event_manager: AutofixEventManager,
        sentry_client: RpcClient = injected,
    ):
        request = state.get().request

        self.organization_id = request.organization_id
        self.project_id = request.project_id
        self.repos = request.repos

        self.sentry_client = sentry_client

        self.event_manager = event_manager
        self.state = state

        logger.info(f"AutofixContext initialized with run_id {self.run_id}")

    @classmethod
    def from_run_id(cls, run_id: int):
        state = ContinuationState(run_id)
        with state.update() as cur:
            cur.mark_triggered()

        event_manager = AutofixEventManager(state)

        return cls(state, event_manager)

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

    def get_issue_summary(self) -> IssueSummaryWithScores | None:
        group_id = self.state.get().request.issue.id
        with Session() as session:
            group_summary = session.get(DbIssueSummary, group_id)
            if group_summary:
                try:
                    return IssueSummaryWithScores.from_db_state(group_summary)
                except ValidationError:
                    return None
        return None

    def repos_by_key(self) -> Mapping[RepoKey, RepoDefinition]:
        repos_by_key: dict[RepoKey, RepoDefinition] = {
            repo.external_id: repo for repo in self.repos
        }

        return repos_by_key

    def get_repo_client(
        self,
        repo_name: str | None = None,
        repo_external_id: str | None = None,
        type: RepoClientType = RepoClientType.READ,
    ) -> RepoClient:
        return get_repo_client(
            repos=self.repos, repo_name=repo_name, repo_external_id=repo_external_id, type=type
        )

    def autocorrect_repo_name(self, repo_name: str) -> str | None:
        return autocorrect_repo_name(
            readable_repos=self.state.get().readable_repos, repo_name=repo_name
        )

    def get_file_contents(
        self, path: str, repo_name: str | None = None, ignore_local_changes: bool = False
    ) -> str | None:
        if len(self.repos) > 1:
            if not repo_name:
                raise ValueError("Repo name is required when there are multiple repos.")

            if repo_name not in [repo.full_name for repo in self.repos]:
                raise ValueError(f"Repo '{repo_name}' not found in the list of repos.")

        repo_client = self.get_repo_client(repo_name)
        file_contents, _ = repo_client.get_file_content(path)

        if not ignore_local_changes:
            cur_state = self.state.get()
            repo_file_changes = cur_state.codebases[repo_client.repo_external_id].file_changes
            current_file_changes = list(filter(lambda x: x.path == path, repo_file_changes))
            for file_change in current_file_changes:
                file_contents = file_change.apply(file_contents)

        return file_contents

    def does_file_exist(
        self, path: str, repo_name: str | None = None, ignore_local_changes: bool = False
    ) -> bool:
        if len(self.repos) > 1:
            if not repo_name:
                raise ValueError("Repo name is required when there are multiple repos.")

            if repo_name not in [repo.full_name for repo in self.repos]:
                raise ValueError(f"Repo '{repo_name}' not found in the list of repos.")

        repo_client = self.get_repo_client(repo_name)
        does_exist_on_remote = repo_client.does_file_exist(path)
        if does_exist_on_remote:
            return True

        if not ignore_local_changes:
            cur_state = self.state.get()
            repo_file_changes = cur_state.codebases[repo_client.repo_external_id].file_changes
            current_file_changes = list(
                filter(lambda x: x.path == path and x.change_type == "create", repo_file_changes)
            )
            if current_file_changes:
                return True

        return False

    def get_commit_history_for_file(
        self, path: str, repo_name: str | None = None, max_commits: int = 10
    ) -> list[str]:
        repo_name = self.autocorrect_repo_name(repo_name) if repo_name else None
        if not repo_name:
            raise AgentError() from ValueError(
                f"Repo '{repo_name}' not found. Available repos: {', '.join([repo.full_name for repo in self.repos])}"
            )

        repo_client = self.get_repo_client(repo_name)
        return repo_client.get_commit_history(path, autocorrect=True, max_commits=max_commits)

    def get_commit_patch_for_file(
        self, path: str, repo_name: str | None = None, commit_sha: str | None = None
    ) -> str | None:
        repo_name = self.autocorrect_repo_name(repo_name) if repo_name else None
        if not repo_name:
            raise AgentError() from ValueError(
                f"Repo '{repo_name}' not found. Available repos: {', '.join([repo.full_name for repo in self.repos])}"
            )

        try:
            repo_client = self.get_repo_client(repo_name)
            return repo_client.get_commit_patch_for_file(path, commit_sha, autocorrect=True)
        except UnknownObjectException as e:
            logger.warning(f"Invalid commit SHA provided: {commit_sha}. Error: {e}")
            return None

    def autocorrect_file_path(
        self, path: str, repo_name: str, files_only: bool = False
    ) -> str | None:
        """
        Attempts to fix a path by checking if it exists in the repository as a path or directory.

        Args:
            path: The path to autocorrect
            repo_name: The name of the repository to use for validation
            files_only: If True, only return valid file paths, not directory paths
        """
        repo_client = self.get_repo_client(repo_name=repo_name)
        all_files = repo_client.get_valid_file_paths()

        normalized_path = path.lstrip("./").lstrip("/")
        if not normalized_path:
            return None

        for p in all_files:
            if p.endswith(normalized_path):
                # is a valid file path
                return p
            if not files_only and p.startswith(normalized_path):
                # is a valid directory path
                return normalized_path

        return None

    def _process_stacktrace_paths(self, stacktrace: Stacktrace):
        """
        Annotate a stacktrace with the correct repo each frame is pointing to and fix the filenames
        """
        supported_repos = [
            repo for repo in self.repos if repo.provider in RepoClient.supported_providers
        ]

        repo_data: dict[str, Tuple[RepoClient, Set[str]]] = {}

        def get_repo_data(repo):
            try:
                repo_client = self.get_repo_client(
                    repo_external_id=repo.external_id, type=RepoClientType.READ
                )
                valid_file_paths = repo_client.get_valid_file_paths()
                return repo.external_id, (repo_client, valid_file_paths)
            except UnknownObjectException:
                self.event_manager.on_error(
                    error_msg=f"Autofix does not have access to the `{repo.full_name}` repo. Please give permission through the Sentry GitHub integration, or remove the repo from your code mappings.",
                    should_completely_error=True,
                )
                return None

        with ThreadPoolExecutor(initializer=copy_modules_initializer()) as executor:
            future_to_repo = {
                executor.submit(get_repo_data, repo): repo for repo in supported_repos
            }

            for future in as_completed(future_to_repo):
                result = future.result()
                if result:
                    repo_external_id, data = result
                    repo_data[repo_external_id] = data

        for frame in stacktrace.frames:
            if frame.in_app and frame.repo_name is None:
                for repo in supported_repos:
                    if repo.external_id not in repo_data:
                        continue

                    _, valid_file_paths = repo_data[repo.external_id]

                    if frame.filename in valid_file_paths:
                        frame.repo_name = repo.full_name
                        break
                    else:
                        for valid_path in valid_file_paths:
                            if potential_frame_match(valid_path, frame):
                                frame.repo_name = repo.full_name
                                frame.filename = valid_path
                                break
                        if frame.repo_name:
                            break

    def process_event_paths(self, event: EventDetails):
        """
        Annotate exceptions with the correct repo each frame is pointing to and fix the filenames
        """
        for exception in event.exceptions:
            if exception.stacktrace:
                self._process_stacktrace_paths(exception.stacktrace)
        for thread in event.threads:
            if thread.stacktrace:
                self._process_stacktrace_paths(thread.stacktrace)

    def _get_change_state(self, repo_external_id: str) -> CodebaseChange | None:
        changes_step = self.state.get().changes_step

        if not changes_step:
            raise ValueError("Changes step not found")

        change_state = next(
            (
                (change)
                for change in changes_step.changes
                if change.repo_external_id == repo_external_id
            ),
            None,
        )

        return change_state

    def _set_change_state(self, change_state: CodebaseChange):
        with self.state.update() as cur:
            changes_step = cur.changes_step

            if not changes_step:
                raise ValueError("Changes step not found")

            changes_state_index = next(
                (
                    i
                    for i, c in enumerate(changes_step.changes)
                    if c.repo_external_id == change_state.repo_external_id
                ),
                None,
            )

            if changes_state_index is None:
                raise ValueError("Change state not found")

            changes_step.changes[changes_state_index] = change_state

    def commit_changes(
        self,
        repo_external_id: str | None = None,
        make_pr: bool = False,
        pr_to_comment_on_url: str | None = None,
    ):
        state = self.state.get()

        root_cause_step = state.root_cause_step
        root_cause_summary = (
            root_cause_step.causes[0].description
            if root_cause_step and root_cause_step.causes
            else None
        )

        for codebase_state in state.codebases.values():
            if repo_external_id is None or codebase_state.repo_external_id == repo_external_id:
                if not codebase_state.repo_external_id:
                    raise ValueError("Repo external ID not found")

                change_state = self._get_change_state(codebase_state.repo_external_id)

                if codebase_state.file_changes and change_state:
                    key = codebase_state.repo_external_id

                    if key is None:
                        raise ValueError("Repo key not found")

                    repo_definition = self.repos_by_key().get(key)

                    if repo_definition is None:
                        raise ValueError(f"Repo definition not found for key {key}")

                    repo_client = self.get_repo_client(
                        repo_external_id=repo_definition.external_id, type=RepoClientType.WRITE
                    )

                    # Because the GitHub API is slow, if we kill this task with a half-created branch, subsequent tries will create new branches.
                    # However, if one already exists and is created we will use it.
                    if not change_state.branch_name:
                        branch_ref = repo_client.create_branch_from_changes(
                            pr_title=change_state.title,
                            file_patches=change_state.diff,
                            branch_name=change_state.draft_branch_name
                            or f"seer/{change_state.title}",
                            from_base_sha=True,  # Autofix PRs are always created from the base commit
                        )

                        if branch_ref is None:
                            logger.warning("Failed to create branch from changes")
                            return None

                        change_state.branch_name = branch_ref.ref.replace("refs/heads/", "")
                        self._set_change_state(change_state)

                    if not make_pr:
                        return

                    change_state = self._get_change_state(codebase_state.repo_external_id)

                    if not change_state:
                        raise ValueError("Change state not found for PR creation")

                    if change_state.pull_request:
                        logger.info(
                            f"Pull request already exists for change in repo {repo_external_id}"
                        )
                        return

                    if not change_state.branch_name:
                        raise ValueError("Branch name not found for PR creation")

                    branch_ref = repo_client.get_branch_ref(change_state.branch_name)

                    if not branch_ref:
                        raise ValueError("Branch not found for PR creation")

                    ref_note = ""
                    org_slug = self.get_org_slug(state.request.organization_id)
                    if org_slug:
                        issue_url = f"https://sentry.io/organizations/{org_slug}/issues/{state.request.issue.id}/"
                        issue_link = (
                            f"[{state.request.issue.short_id}]({issue_url})"
                            if state.request.issue.short_id
                            else issue_url
                        )
                        suspect_pr_link = (
                            f", which was likely introduced in [this PR]({pr_to_comment_on_url})."
                            if pr_to_comment_on_url
                            else ""
                        )
                        ref_note = f"Fixes {issue_link}{suspect_pr_link}."

                    pr_description = textwrap.dedent(
                        """\
                        {ref_note} {root_cause}

                        {description}

                        {user_line}👁️ Run ID: {run_id}"""
                    ).format(
                        run_id=state.run_id,
                        user_line=(
                            f"\nThis fix was generated by Seer in Sentry, triggered by {state.request.invoking_user.display_name}. "
                            if state.request.invoking_user
                            else ""
                        ),
                        description=change_state.description,
                        root_cause=(
                            f"The issue was that: {root_cause_summary}"
                            if root_cause_summary
                            else ""
                        ),
                        ref_note=ref_note,
                    )

                    pr = repo_client.create_pr_from_branch(
                        branch_ref, change_state.title, pr_description
                    )

                    change_state.pull_request = CommittedPullRequestDetails(
                        pr_number=pr.number, pr_url=pr.html_url, pr_id=pr.id
                    )

                    self._set_change_state(change_state)

                    with Session() as session:
                        pr_id_mapping = DbPrIdToAutofixRunIdMapping(
                            provider=repo_client.provider,
                            pr_id=pr.id,
                            run_id=state.run_id,
                        )
                        session.add(pr_id_mapping)
                        session.commit()

                    if (
                        pr_to_comment_on_url
                    ):  # for GitHub Copilot, leave a comment that the PR is made
                        repo_client.comment_pr_generated_for_copilot(
                            pr_to_comment_on_url=pr_to_comment_on_url,
                            new_pr_url=pr.html_url,
                            run_id=state.run_id,
                        )

    def comment_root_cause_on_pr(
        self, pr_url: str, repo_definition: RepoDefinition, root_cause: str
    ):
        # make root cause into markdown string
        state = self.state.get()
        markdown_comment = textwrap.dedent(
            """\
            👋 Hi there! Here is a root cause analysis of {issue} automatically generated by Autofix 🤖
            {user_line}

            {root_cause}

            ## More Info

            If you have any questions or feedback for the Sentry team about this fix, please email [autofix@sentry.io](mailto:autofix@sentry.io) with the Run ID: {run_id}."""
        ).format(
            run_id=state.run_id,
            issue=(
                f"issue {state.request.issue.short_id}"
                if state.request.issue.short_id
                else "the issue above"
            ),
            user_line=(
                f"\nThis analysis was triggered by {state.request.invoking_user.display_name}."
                if state.request.invoking_user
                else ""
            ),
            root_cause=root_cause,
        )

        # comment root cause analysis on PR
        repo_client = self.get_repo_client(
            repo_external_id=repo_definition.external_id, type=RepoClientType.READ
        )
        repo_client.comment_root_cause_on_pr_for_copilot(
            pr_url, state.run_id, state.request.issue.id, markdown_comment
        )

    @functools.lru_cache(maxsize=8)
    def get_org_slug(self, organization_id: int) -> str | None:
        slug: str | None = None
        try:
            response = self.sentry_client.call("get_organization_slug", org_id=organization_id)
            slug = None if response is None else response.get("slug", None)
            if slug is None:
                logger.warn(
                    f"Slug lookup call for organization {organization_id} succeeded but returned value None."
                )
        except Exception as e:
            logger.warn(f"Failed to get slug for organization {organization_id}")
            logger.exception(e)
            sentry_sdk.capture_exception(e)
            slug = None
        return slug

    def make_file_patches(
        self, file_changes: list[FileChange], repo_name: str
    ) -> tuple[list[FilePatch], str]:
        changes_by_path: dict[str, list[FileChange]] = {}
        for change in file_changes:
            if change.path not in changes_by_path:
                changes_by_path[change.path] = []
            changes_by_path[change.path].append(change)

        if not changes_by_path.keys():
            return [], ""

        original_documents = []
        for path, changes in changes_by_path.items():
            is_new_file = all(change.change_type == "create" for change in changes)
            if not is_new_file:
                original_documents.append(
                    BaseDocument(
                        path=path,
                        text=self.get_file_contents(
                            path, repo_name=repo_name, ignore_local_changes=True
                        )
                        or "",
                    )
                )
            else:
                original_documents.append(
                    BaseDocument(
                        path=path,
                        text="",
                    )
                )

        return make_file_patches(file_changes, list(changes_by_path.keys()), original_documents)

    def store_memory(self, key: str, memory: list[Message]):
        with Session() as session:
            memory_record = (
                session.query(DbRunMemory).where(DbRunMemory.run_id == self.run_id).one_or_none()
            )

            if not memory_record:
                memory_model = AutofixRunMemory(run_id=self.run_id)
            else:
                memory_model = AutofixRunMemory.from_db_model(memory_record)

            memory_model.memory[key] = memory
            memory_record = memory_model.to_db_model()

            session.merge(memory_record)
            session.commit()

    def get_memory(self, key: str) -> list[Message]:
        with Session() as session:
            memory_record = (
                session.query(DbRunMemory).where(DbRunMemory.run_id == self.run_id).one_or_none()
            )

            if not memory_record:
                return []

            return AutofixRunMemory.from_db_model(memory_record).memory.get(key, [])
