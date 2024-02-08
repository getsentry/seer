import logging
import os
import textwrap
from typing import List

import requests
import sentry_sdk
from github import Auth, Github, GithubIntegration
from github.GitRef import GitRef
from github.Repository import Repository
from unidiff import PatchSet

from seer.automation.agent.types import Usage
from seer.automation.autofix.models import FileChange, PlanStep
from seer.automation.autofix.utils import generate_random_string, sanitize_branch_name

logger = logging.getLogger("autofix")


def get_github_auth(repo_owner: str, repo_name: str):
    app_id = os.environ.get("GITHUB_APP_ID")
    private_key = os.environ.get("GITHUB_PRIVATE_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")

    if github_token is None and (app_id is None or private_key is None):
        raise ValueError(
            "Need either GITHUB_TOKEN or (GITHUB_APP_ID and GITHUB_PRIVATE_KEY) to be set."
        )

    github_auth: Auth.Token | Auth.AppInstallationAuth
    if github_token is not None:
        github_auth = Auth.Token(github_token)
    else:
        app_auth = Auth.AppAuth(app_id, private_key=private_key)  # type: ignore
        gi = GithubIntegration(auth=app_auth)
        installation = gi.get_repo_installation(repo_owner, repo_name)
        github_auth = app_auth.get_installation_auth(installation.id)

    return github_auth


class RepoClient:
    # TODO: Support other git providers later
    github_auth: Auth.Token | Auth.AppInstallationAuth
    github: Github
    repo: Repository
    base_sha: str

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
    ):
        self.github = Github(auth=get_github_auth(repo_owner, repo_name))
        self.repo = self.github.get_repo(repo_owner + "/" + repo_name)

        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def get_commit_file_diffs(self, prev_sha: str, next_sha: str) -> tuple[list[str], list[str]]:
        """
        Returns the list of files to change and files to delete in the diff in order to turn a commit into another.
        """
        comparison = self.repo.compare(prev_sha, next_sha)

        # Support reverse diffs, because the api would return an empty list of files if the comparison is behind
        is_behind = comparison.status == "behind"
        if is_behind:
            comparison = self.repo.compare(next_sha, prev_sha)

        # Hack: We're extracting the authorization and user agent headers from the PyGithub library to get this diff
        # This has to be done because the files list inside the comparison object is limited to only 300 files.
        # We get the entire diff from the diff object returned from the `diff_url`
        requester = self.repo._requester
        headers = {
            "Authorization": f"{requester._Requester__auth.token_type} {requester._Requester__auth.token}",  # type: ignore
            "User-Agent": requester._Requester__userAgent,  # type: ignore
        }
        data = requests.get(comparison.diff_url, headers=headers).content

        patch_set = PatchSet(data.decode("utf-8"))

        added_files = [patch.path for patch in patch_set.added_files]
        modified_files = [patch.path for patch in patch_set.modified_files]
        removed_files = [patch.path for patch in patch_set.removed_files]

        if is_behind:
            # If the comparison is behind, the added files are actually the removed files
            changed_files = list(set(modified_files + removed_files))
            removed_files = added_files
        else:
            changed_files = list(set(added_files + modified_files))

        return changed_files, removed_files

    def _create_branch(self, branch_name, base_branch=None, base_commit_sha: str | None = None):
        base_sha = base_commit_sha
        if base_branch:
            base_sha = self.repo.get_branch(base_branch).commit.sha

        if not base_sha:
            raise ValueError("base_sha cannot be None")

        ref = self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
        return ref

    def _commit_file_change(self, change: FileChange, branch_ref: str):
        contents = self.repo.get_contents(change.path, ref=branch_ref)

        if isinstance(contents, list):
            raise Exception(f"Expected a single ContentFile but got a list for path {change.path}")

        new_contents = change.apply(contents.decoded_content.decode("utf-8") if contents else None)

        if new_contents is None:
            self.repo.delete_file(
                change.path, change.description or "File deletion", contents.sha, branch=branch_ref
            )
        else:
            self.repo.update_file(
                change.path,
                change.description or "File change",
                new_contents,
                contents.sha,
                branch=branch_ref,
            )

    def create_branch_from_changes(
        self, pr_title: str, file_changes: list[FileChange], base_commit_sha: str
    ) -> GitRef | None:
        new_branch_name = f"autofix/{sanitize_branch_name(pr_title)}/{generate_random_string(n=6)}"
        branch_ref = self._create_branch(new_branch_name, base_commit_sha=base_commit_sha)

        for change in file_changes:
            try:
                self._commit_file_change(change, branch_ref.ref)
            except Exception as e:
                logger.error(f"Error committing file change: {e}")

        branch_ref.update()

        # Check that the changes were made
        comparison = self.repo.compare(base_commit_sha, branch_ref.object.sha)

        if comparison.ahead_by < 1:
            # Remove the branch if there are no changes
            try:
                self.repo.get_git_ref(branch_ref.ref).delete()
            except github.UnknownObjectException:
                logger.warning("Attempted to delete a branch with no commits.")
            sentry_sdk.capture_message(
                f"Failed to create branch from changes. Comparison is ahead by {comparison.ahead_by}"
            )
            return None

        return branch_ref

    def _get_stats_str(self, prompt_tokens: int, completion_tokens: int):
        stats_str = textwrap.dedent(
            f"""\
            Prompt tokens: **{prompt_tokens}**
            Completion tokens: **{completion_tokens}**
            Total tokens: **{prompt_tokens + completion_tokens}**"""
        )

        return stats_str

    def create_pr_from_branch(
        self,
        branch: GitRef,
        title: str,
        description: str,
        steps: list[PlanStep],
        issue_id: int | str,
        usage: Usage,
    ):
        title = f"""🤖 {title}"""

        issue_link = f"https://sentry.io/organizations/sentry/issues/{issue_id}/"

        description = textwrap.dedent(
            """\
            👋 Hi there! This PR was automatically generated 🤖

            {description}

            #### The steps that were performed:
            {steps}

            #### The issue that triggered this PR:
            {issue_link}

            ### 📣 Instructions for the reviewer which is you, yes **you**:
            - **If these changes were incorrect, please close this PR and comment explaining why.**
            - **If these changes were incomplete, please continue working on this PR then merge it.**
            - **If you are feeling confident in my changes, please merge this PR.**

            This will greatly help us improve the autofix system. Thank you! 🙏

            If there are any questions, please reach out to the [AI/ML Team](https://github.com/orgs/getsentry/teams/machine-learning-ai) on [#proj-suggested-fix](https://sentry.slack.com/archives/C06904P7Z6E)

            ### 🤓 Stats for the nerds:
            {stats}"""
        ).format(
            description=description,
            issue_link=issue_link,
            steps="\n".join([f"{i + 1}. {step.title}" for i, step in enumerate(steps)]),
            stats=self._get_stats_str(usage.prompt_tokens, usage.completion_tokens),
        )

        return self.repo.create_pull(
            title=title, body=description, base="master", head=branch.ref, draft=True
        )
