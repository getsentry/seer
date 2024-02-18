import logging
import os
import shutil
import tarfile
import tempfile
import textwrap

import requests
import sentry_sdk
from github import Auth, Github, GithubIntegration, UnknownObjectException
from github.GitRef import GitRef
from github.Repository import Repository
from unidiff import PatchSet

from seer.automation.agent.models import Usage
from seer.automation.autofix.models import FileChange, PlanStep
from seer.automation.autofix.utils import generate_random_string, sanitize_branch_name
from seer.automation.models import InitializationError
from seer.utils import class_method_lru_cache

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

    provider: str

    def __init__(
        self,
        repo_provider: str,
        repo_owner: str,
        repo_name: str,
    ):
        if repo_provider != "github":
            # This should never get here, the repo provider should be checked on the Sentry side but this will make debugging
            # easier if it does
            raise InitializationError(
                f"Unsupported repo provider: {repo_provider}, only github is supported."
            )

        self.provider = repo_provider
        self.github = Github(auth=get_github_auth(repo_owner, repo_name))
        self.repo = self.github.get_repo(repo_owner + "/" + repo_name)

        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def get_default_branch(self):
        return self.repo.default_branch

    def get_default_branch_head_sha(self):
        return self.repo.get_branch(self.get_default_branch()).commit.sha

    def compare(self, base: str, head: str):
        return self.repo.compare(base, head)

    def load_repo_to_tmp_dir(self, sha: str) -> tuple[str, str]:
        # Check if output directory exists, if not create it
        tmp_dir = tempfile.mkdtemp(prefix=f"{self.repo_owner}-{self.repo_name}_{sha}")
        tmp_repo_dir = os.path.join(tmp_dir, "repo")

        logger.debug(f"Loading repository to {tmp_repo_dir}")

        # Create a temporary directory to store the repository
        os.makedirs(tmp_repo_dir, exist_ok=True)

        # Clean the directory
        for root, dirs, files in os.walk(tmp_repo_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        tarball_url = self.repo.get_archive_link("tarball", ref=sha)
        tarfile_path = os.path.join(tmp_dir, f"{sha}.tar.gz")

        response = requests.get(tarball_url, stream=True)
        if response.status_code == 200:
            with open(tarfile_path, "wb") as f:
                f.write(response.content)
        else:
            logger.error(
                f"Failed to get tarball url for {tarball_url}. Please check if the repository exists and the provided token is valid."
            )
            logger.error(
                f"Response status code: {response.status_code}, response text: {response.text}"
            )
            raise Exception(
                f"Failed to get tarball url for {tarball_url}. Please check if the repository exists and the provided token is valid."
            )

        # Extract tarball into the output directory
        with tarfile.open(tarfile_path, "r:gz") as tar:
            tar.extractall(path=tmp_repo_dir)  # extract all members normally
            extracted_folders = [
                name
                for name in os.listdir(tmp_repo_dir)
                if os.path.isdir(os.path.join(tmp_repo_dir, name))
            ]
            if extracted_folders:
                root_folder = extracted_folders[0]  # assuming the first folder is the root folder
                root_folder_path = os.path.join(tmp_repo_dir, root_folder)
                for item in os.listdir(root_folder_path):
                    s = os.path.join(root_folder_path, item)
                    d = os.path.join(tmp_repo_dir, item)
                    if os.path.isdir(s):
                        shutil.move(
                            s, d
                        )  # move all directories from the root folder to the output directory
                    else:
                        shutil.copy2(
                            s, d
                        )  # copy all files from the root folder to the output directory
                shutil.rmtree(root_folder_path)  # remove the root folder

        return tmp_dir, tmp_repo_dir

    @class_method_lru_cache(maxsize=16)
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

    def get_file_content(self, path: str, sha: str) -> str | None:
        logger.debug(f"Getting file contents for {path} in {self.repo.full_name} on sha {sha}")
        try:
            contents = self.repo.get_contents(path, ref=sha)

            if isinstance(contents, list):
                raise Exception(f"Expected a single ContentFile but got a list for path {path}")

            return contents.decoded_content.decode()
        except Exception as e:
            logger.error(f"Error getting file contents: {e}")

            return None

    def get_valid_file_paths(self, sha: str) -> set[str]:
        tree = self.repo.get_git_tree(sha, recursive=True)

        if tree.raw_data["truncated"]:
            sentry_sdk.capture_message(
                f"Truncated tree for {self.repo.full_name}. This may cause issues with autofix."
            )

        valid_file_paths: set[str] = set()

        for file in tree.tree:
            valid_file_paths.add(file.path)

        return valid_file_paths

    def _create_branch(self, branch_name):
        ref = self.repo.create_git_ref(
            ref=f"refs/heads/{branch_name}", sha=self.get_default_branch_head_sha()
        )

        return ref

    def _commit_file_change(self, change: FileChange, branch_ref: str):
        contents = self.repo.get_contents(change.path, ref=branch_ref)

        if isinstance(contents, list):
            raise Exception(f"Expected a single ContentFile but got a list for path {change.path}")

        new_contents = change.apply(contents.decoded_content.decode("utf-8") if contents else None)

        # TODO: Do this elsewhere
        if change.path.startswith("/"):
            change.path = change.path[1:]

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
        self, pr_title: str, file_changes: list[FileChange]
    ) -> GitRef | None:
        new_branch_name = f"autofix/{sanitize_branch_name(pr_title)}/{generate_random_string(n=6)}"
        branch_ref = self._create_branch(new_branch_name)

        for change in file_changes:
            try:
                self._commit_file_change(change, branch_ref.ref)
            except Exception as e:
                logger.error(f"Error committing file change: {e}")

        branch_ref.update()

        # Check that the changes were made
        comparison = self.repo.compare(self.get_default_branch_head_sha(), branch_ref.object.sha)

        if comparison.ahead_by < 1:
            # Remove the branch if there are no changes
            try:
                branch_ref.delete()
            except UnknownObjectException:
                logger.warning("Attempted to delete a branch or reference that does not exist.")
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
            title=title,
            body=description,
            base=self.get_default_branch(),
            head=branch.ref,
            draft=True,
        )
