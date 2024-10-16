import logging
import os
import shutil
import tarfile
import tempfile
from enum import Enum
from typing import Literal

import requests
import sentry_sdk
from github import Auth, Github, GithubIntegration, UnknownObjectException
from github.GitRef import GitRef
from github.Repository import Repository
from unidiff import PatchSet

from seer.automation.autofix.utils import generate_random_string, sanitize_branch_name
from seer.automation.codebase.utils import get_language_from_path
from seer.automation.models import FileChange, InitializationError, RepoDefinition
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected
from seer.utils import class_method_lru_cache

logger = logging.getLogger(__name__)


def get_github_app_auth_and_installation(
    app_id: int | str, private_key: str, repo_owner: str, repo_name: str
):
    app_auth = Auth.AppAuth(app_id, private_key=private_key)
    gi = GithubIntegration(auth=app_auth)
    installation = gi.get_repo_installation(repo_owner, repo_name)
    github_auth = app_auth.get_installation_auth(installation.id)

    return github_auth, installation


def get_repo_app_permissions(
    app_id: int | str, private_key: str, repo_owner: str, repo_name: str
) -> dict[str, str] | None:
    try:
        _, installation = get_github_app_auth_and_installation(
            app_id, private_key, repo_owner, repo_name
        )

        return installation.raw_data.get("permissions", {})
    except UnknownObjectException:
        return None


@inject
def get_github_token_auth(config: AppConfig = injected) -> Auth.Token | None:
    github_token = config.GITHUB_TOKEN
    if github_token is None:
        return None

    return Auth.Token(github_token)


@inject
def get_write_app_credentials(config: AppConfig = injected) -> tuple[int | str | None, str | None]:
    app_id = config.GITHUB_APP_ID
    private_key = config.GITHUB_PRIVATE_KEY

    if not app_id or not private_key:
        return None, None

    return app_id, private_key


@inject
def get_read_app_credentials(config: AppConfig = injected) -> tuple[int | str | None, str | None]:
    app_id = config.GITHUB_SENTRY_APP_ID
    private_key = config.GITHUB_SENTRY_PRIVATE_KEY

    if not app_id or not private_key:
        return get_write_app_credentials()

    return app_id, private_key


@inject
def get_codecov_unit_test_app_credentials(
    config: AppConfig = injected,
) -> tuple[int | str | None, str | None]:
    app_id = config.GITHUB_CODECOV_UNIT_TEST_APP_ID
    private_key = config.GITHUB_CODECOV_UNIT_TEST_PRIVATE_KEY

    if not app_id or not private_key:
        return get_write_app_credentials()

    return app_id, private_key


class RepoClientType(str, Enum):
    READ = "read"
    WRITE = "write"
    CODECOV_UNIT_TEST = "codecov_unit_test"


class RepoClient:
    # TODO: Support other git providers later
    github_auth: Auth.Token | Auth.AppInstallationAuth
    github: Github
    repo: Repository

    provider: str
    repo_owner: str
    repo_name: str
    repo_external_id: str
    base_commit_sha: str

    def __init__(
        self, app_id: int | str | None, private_key: str | None, repo_definition: RepoDefinition
    ):
        if repo_definition.provider != "github":
            # This should never get here, the repo provider should be checked on the Sentry side but this will make debugging
            # easier if it does
            raise InitializationError(
                f"Unsupported repo provider: {repo_definition.provider}, only github is supported."
            )

        if app_id and private_key:
            self.github = Github(
                auth=get_github_app_auth_and_installation(
                    app_id, private_key, repo_definition.owner, repo_definition.name
                )[0]
            )
        else:
            self.github = Github(auth=get_github_token_auth())

        self.repo = self.github.get_repo(
            int(repo_definition.external_id)
            if repo_definition.external_id.isdigit()
            else repo_definition.full_name
        )

        self.provider = repo_definition.provider
        self.repo_owner = repo_definition.owner
        self.repo_name = repo_definition.name
        self.repo_external_id = repo_definition.external_id
        self.base_commit_sha = repo_definition.base_commit_sha or self.get_default_branch_head_sha()

    @staticmethod
    def check_repo_write_access(repo: RepoDefinition):
        app_id, pk = get_write_app_credentials()

        if app_id is None or pk is None:
            return True if get_github_token_auth() else None

        permissions = get_repo_app_permissions(app_id, pk, repo.owner, repo.name)

        if (
            permissions
            and permissions.get("contents") == "write"
            and permissions.get("pull_requests") == "write"
        ):
            return True

        return False

    @staticmethod
    def check_repo_read_access(repo: RepoDefinition):
        app_id, pk = get_write_app_credentials()

        if app_id is None or pk is None:
            return True if get_github_token_auth() else None

        permissions = get_repo_app_permissions(app_id, pk, repo.owner, repo.name)

        if permissions and (
            permissions.get("contents") == "read" or permissions.get("contents") == "write"
        ):
            return True

        return False

    @classmethod
    def from_repo_definition(cls, repo_def: RepoDefinition, type: RepoClientType):
        if type == RepoClientType.WRITE:
            return cls(*get_write_app_credentials(), repo_def)
        elif type == RepoClientType.CODECOV_UNIT_TEST:
            return cls(*get_codecov_unit_test_app_credentials(), repo_def)

        return cls(*get_read_app_credentials(), repo_def)

    @property
    def repo_full_name(self):
        return self.repo.full_name

    def get_default_branch(self) -> str:
        return self.repo.default_branch

    def get_branch_head_sha(self, branch: str):
        return self.repo.get_branch(branch).commit.sha

    def get_default_branch_head_sha(self):
        return self.get_branch_head_sha(self.get_default_branch())

    def compare(self, base: str, head: str):
        return self.repo.compare(base, head)

    def load_repo_to_tmp_dir(self, sha: str | None = None) -> tuple[str, str]:
        sha = sha or self.base_commit_sha

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
                    # TODO: Consider a strategy for handling symlinks more appropriately in the future, possibly by resolving them or copying as symlinks to maintain the original structure.
                    if os.path.isdir(s):
                        shutil.move(
                            s, d
                        )  # move all directories from the root folder to the output directory
                    else:
                        # Skipping symlinks to prevent FileNotFoundError.
                        if not os.path.islink(s):
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

        data = requests.get(comparison.diff_url, headers=self._get_auth_headers(accept_type="diff"))
        data.raise_for_status()  # Raise an exception for HTTP errors

        patch_set = PatchSet(data.content.decode("utf-8"))

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

    def get_file_content(self, path: str, sha: str | None = None) -> str | None:
        logger.debug(f"Getting file contents for {path} in {self.repo.full_name} on sha {sha}")
        if sha is None:
            sha = self.base_commit_sha
        try:
            contents = self.repo.get_contents(path, ref=sha)

            if isinstance(contents, list):
                raise Exception(f"Expected a single ContentFile but got a list for path {path}")

            return contents.decoded_content.decode()
        except Exception as e:
            logger.error(f"Error getting file contents: {e}")

            return None

    def get_valid_file_paths(self, sha: str | None = None) -> set[str]:
        if sha is None:
            sha = self.base_commit_sha

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
        contents = (
            self.repo.get_contents(change.path, ref=branch_ref)
            if change.change_type != "create"
            else None
        )

        if isinstance(contents, list):
            raise RuntimeError(
                f"Expected a single ContentFile but got a list for path {change.path}"
            )

        new_contents = change.apply(contents.decoded_content.decode("utf-8") if contents else None)

        # Remove leading slash if it exists, the github api will reject paths with leading slashes.
        if change.path.startswith("/"):
            change.path = change.path[1:]

        if change.change_type == "delete" and contents:
            self.repo.delete_file(
                change.path,
                change.description or "File deletion",
                contents.sha,  # FYI: It wants the sha of the content blob here, not a commit sha.
                branch=branch_ref,
            )
        elif change.change_type == "create" and new_contents:
            self.repo.create_file(
                change.path, change.description or "New file", new_contents, branch=branch_ref
            )
        else:
            if contents is None:
                raise FileNotFoundError(f"File {change.path} does not exist in the repository.")

            self.repo.update_file(
                change.path,
                change.commit_message or "File change",
                new_contents or "",
                contents.sha,  # FYI: It wants the sha of the content blob here, not a commit sha.
                branch=branch_ref,
            )

    def create_branch_from_changes(
        self, pr_title: str, file_changes: list[FileChange], branch_name: str | None = None
    ) -> GitRef | None:
        new_branch_name = (
            branch_name or f"autofix/{sanitize_branch_name(pr_title)}/{generate_random_string(n=6)}"
        )
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

    def create_pr_from_branch(
        self,
        branch: GitRef,
        title: str,
        description: str,
        provided_base: str | None = None,
    ):
        return self.repo.create_pull(
            title=title,
            body=description,
            base=provided_base or self.get_default_branch(),
            head=branch.ref,
            draft=True,
        )

    def get_index_file_set(
        self, sha: str | None = None, max_file_size_bytes=2 * 1024 * 1024, skip_empty_files=False
    ) -> set[str]:
        if sha is None:
            sha = self.base_commit_sha

        tree = self.repo.get_git_tree(sha=sha, recursive=True)

        # Recursive tree requests are truncated at 100,000 entries or 7MB as noted @ https://docs.github.com/en/rest/git/trees?apiVersion=2022-11-28#get-a-tree
        # This should be sufficient for most repositories, but if it's not, we should consider paginating the tree.
        # We log to see how often this happens and if it's a problem.
        if tree.raw_data["truncated"]:
            sentry_sdk.capture_message(
                f"Truncated tree for {self.repo.full_name}. This may cause issues with autofix."
            )

        file_set = set()
        for file in tree.tree:
            if (
                file.type == "blob"
                and file.size < max_file_size_bytes
                and file.mode
                in ["100644", "100755"]  # 100644 is a regular file, 100755 is an executable file
                and get_language_from_path(file.path) is not None
                and (not skip_empty_files or file.size > 0)
            ):
                file_set.add(file.path)

        return file_set

    def get_pr_diff_content(self, pr_url: str) -> str:
        data = requests.get(pr_url, headers=self._get_auth_headers(accept_type="diff"))

        data.raise_for_status()  # Raise an exception for HTTP errors
        return data.text

    def _get_auth_headers(self, accept_type: Literal["json", "diff"] = "json"):
        requester = self.repo._requester
        if requester.auth is None:
            raise Exception("No auth token found for GitHub API")
        headers = {
            "Accept": (
                "application/vnd.github.diff"
                if accept_type == "diff"
                else "application/vnd.github+json"
            ),
            "Authorization": f"Bearer {requester.auth.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        return headers

    def comment_root_cause_on_pr_for_copilot(
        self, pr_url: str, run_id: int, issue_id: int, comment: str
    ):
        pull_id = int(pr_url.split("/")[-1])
        repo_name = pr_url.split("github.com/")[1].split("/pull")[0]  # should be "owner/repo"
        url = f"https://api.github.com/repos/{repo_name}/issues/{pull_id}/comments"
        params = {
            "body": comment,
            "actions": [
                {
                    "name": "Fix with Sentry",
                    "type": "copilot-chat",
                    "prompt": f"@sentry find a fix for issue {issue_id} with run ID {run_id}",
                }
            ],
        }
        headers = self._get_auth_headers()
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()

    def comment_pr_generated_for_copilot(
        self, pr_to_comment_on_url: str, new_pr_url: str, run_id: int
    ):
        pull_id = int(pr_to_comment_on_url.split("/")[-1])
        repo_name = pr_to_comment_on_url.split("github.com/")[1].split("/pull")[
            0
        ]  # should be "owner/repo"
        url = f"https://api.github.com/repos/{repo_name}/issues/{pull_id}/comments"

        comment = f"A fix has been generated and is available [here]({new_pr_url}) for your review. Autofix Run ID: {run_id}"

        params = {"body": comment}

        headers = self._get_auth_headers()

        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()

    def get_pr_head_sha(self, pr_url: str) -> str:
        data = requests.get(pr_url, headers=self._get_auth_headers(accept_type="json"))
        data.raise_for_status()  # Raise an exception for HTTP errors
        return data.json()["head"]["sha"]

    def post_unit_test_reference_to_original_pr(self, original_pr_url: str, unit_test_pr_url: str):
        original_pr_id = int(original_pr_url.split("/")[-1])
        repo_name = original_pr_url.split("github.com/")[1].split("/pull")[0]
        url = f"https://api.github.com/repos/{repo_name}/issues/{original_pr_id}/comments"
        comment = f"Sentry has generated a new [PR]({unit_test_pr_url}) with unit tests for this PR. View the new PR({unit_test_pr_url}) to review the changes."
        params = {"body": comment}
        headers = self._get_auth_headers()
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()
        return response.json()["html_url"]
