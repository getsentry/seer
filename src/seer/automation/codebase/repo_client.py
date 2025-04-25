import functools
import logging
import os
import shutil
import tarfile
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Literal

import requests
import sentry_sdk
from github import (
    Auth,
    Github,
    GithubException,
    GithubIntegration,
    GithubObject,
    InputGitTreeElement,
    UnknownObjectException,
)
from github.GitRef import GitRef
from github.GitTree import GitTree
from github.GitTreeElement import GitTreeElement
from github.PullRequest import PullRequest
from github.Repository import Repository

from seer.automation.autofix.utils import generate_random_string, sanitize_branch_name
from seer.automation.codebase.models import GithubPrReviewComment
from seer.automation.codebase.utils import get_all_supported_extensions, get_language_from_path
from seer.automation.models import FileChange, FilePatch, InitializationError, RepoDefinition
from seer.automation.utils import detect_encoding
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

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
        sentry_sdk.capture_message("Invalid credentials for codecov unit test app.")
        return get_write_app_credentials()

    return app_id, private_key


@inject
def get_codecov_pr_review_app_credentials(
    config: AppConfig = injected,
) -> tuple[int | str | None, str | None]:
    app_id = config.GITHUB_CODECOV_PR_REVIEW_APP_ID
    private_key = config.GITHUB_CODECOV_PR_REVIEW_PRIVATE_KEY

    if not app_id:
        logger.warning("No key set GITHUB_CODECOV_PR_REVIEW_APP_ID")
    if not private_key:
        logger.warning("No key set GITHUB_CODECOV_PR_REVIEW_PRIVATE_KEY")

    if not app_id or not private_key:
        sentry_sdk.capture_message("Invalid credentials for codecov pr review app.")
        return get_write_app_credentials()

    return app_id, private_key


class RepoClientType(str, Enum):
    READ = "read"
    WRITE = "write"
    CODECOV_UNIT_TEST = "codecov_unit_test"
    CODECOV_PR_REVIEW = "codecov_pr_review"
    CODECOV_PR_CLOSED = "codecov_pr_closed"


class CompleteGitTree:
    """
    A custom class that mimics the interface of github.GitTree
    but allows combining multiple trees into one complete representation.
    """

    def __init__(self, github_tree: GitTree | None = None) -> None:
        self.tree: List[GitTreeElement] = []
        self.raw_data: Dict[str, Any] = {"truncated": False}

        if github_tree:
            self.add_items(github_tree.tree)
            for key, value in github_tree.raw_data.items():
                if key != "truncated":  # We always set truncated to False for our complete tree
                    self.raw_data[key] = value

    def add_item(self, item: GitTreeElement) -> None:
        """Add a tree item to this collection"""
        self.tree.append(item)

    def add_items(self, items: List[GitTreeElement]) -> None:
        """Add multiple tree items to this collection"""
        self.tree.extend(items)


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
    base_branch: str

    supported_providers = ["github"]

    def __init__(
        self, app_id: int | str | None, private_key: str | None, repo_definition: RepoDefinition
    ):
        if repo_definition.provider not in self.supported_providers:
            # This should never get here, the repo provider should be checked on the Sentry side but this will make debugging
            # easier if it does
            raise InitializationError(
                f"Unsupported repo provider: {repo_definition.provider}, only {', '.join(self.supported_providers)} are supported."
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
        self.base_branch = repo_definition.branch_name or self.get_default_branch()
        self.base_commit_sha = repo_definition.base_commit_sha or self.get_branch_head_sha(
            self.base_branch
        )

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
        app_id, pk = get_read_app_credentials()

        if app_id is None or pk is None:
            return True if get_github_token_auth() else None

        permissions = get_repo_app_permissions(app_id, pk, repo.owner, repo.name)

        if permissions and (
            permissions.get("contents") == "read" or permissions.get("contents") == "write"
        ):
            return True

        return False

    @staticmethod
    def _extract_id_from_pr_url(pr_url: str):
        """
        Extracts the repository path and PR/issue ID from the provided URL.
        """
        pr_id = int(pr_url.split("/")[-1])
        return pr_id

    @classmethod
    @functools.lru_cache(maxsize=8)
    def from_repo_definition(cls, repo_def: RepoDefinition, type: RepoClientType):
        if type == RepoClientType.WRITE:
            return cls(*get_write_app_credentials(), repo_def)
        elif type == RepoClientType.CODECOV_UNIT_TEST:
            return cls(*get_codecov_unit_test_app_credentials(), repo_def)
        elif type in (RepoClientType.CODECOV_PR_REVIEW, RepoClientType.CODECOV_PR_CLOSED):
            return cls(*get_codecov_pr_review_app_credentials(), repo_def)

        return cls(*get_read_app_credentials(), repo_def)

    @property
    def repo_full_name(self):
        return self.repo.full_name

    def get_default_branch(self) -> str:
        return self.repo.default_branch

    def get_branch_head_sha(self, branch: str):
        return self.repo.get_branch(branch).commit.sha

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

    def _autocorrect_path(self, path: str, sha: str | None = None) -> tuple[str, bool]:
        """
        Attempts to autocorrect a file path by finding the closest match in the repository.

        Args:
            path: The path to autocorrect
            sha: The commit SHA to use for finding valid paths

        Returns:
            A tuple of (corrected_path, was_autocorrected)
        """
        if sha is None:
            sha = self.base_commit_sha

        path = path.lstrip("/")
        valid_paths = self.get_valid_file_paths(sha)

        # If path is valid, return it unchanged
        if path in valid_paths:
            return path, False

        # Check for partial matches if no exact match and path is long enough
        if len(path) > 3:
            path_lower = path.lower()
            partial_matches = [
                valid_path for valid_path in valid_paths if path_lower in valid_path.lower()
            ]
            if partial_matches:
                # Sort by length to get closest match (shortest containing path)
                closest_match = sorted(partial_matches, key=len)[0]
                logger.warning(
                    f"Path '{path}' not found exactly, using closest match: '{closest_match}'"
                )
                return closest_match, True

        # No match found
        logger.warning("No matching file found for provided file path", extra={"path": path})
        return path, False

    def get_file_content(
        self, path: str, sha: str | None = None, autocorrect: bool = False
    ) -> tuple[str | None, str]:
        logger.debug(f"Getting file contents for {path} in {self.repo.full_name} on sha {sha}")
        if sha is None:
            sha = self.base_commit_sha

        autocorrected_path = False
        if autocorrect:
            path, autocorrected_path = self._autocorrect_path(path, sha)
            if not autocorrected_path and path not in self.get_valid_file_paths(sha):
                return None, "utf-8"

        try:
            contents = self.repo.get_contents(path, ref=sha)

            if isinstance(contents, list):
                raise Exception(f"Expected a single ContentFile but got a list for path {path}")

            detected_encoding = detect_encoding(contents.decoded_content) if contents else "utf-8"
            try:
                content = contents.decoded_content.decode(detected_encoding)
            except UnicodeDecodeError:
                # fallback to utf-8; may still not work
                detected_encoding = "utf-8"
                content = contents.decoded_content.decode(detected_encoding)
            if autocorrected_path:
                content = f"Showing results instead for {path}\n=====\n{content}"
            return content, detected_encoding
        except Exception as e:
            logger.exception(f"Error getting file contents: {e}")
            return None, "utf-8"

    @functools.lru_cache(maxsize=8)
    def get_valid_file_paths(self, commit_sha: str | None = None) -> set[str]:
        if commit_sha is None:
            commit_sha = self.base_commit_sha

        tree = self.get_git_tree(commit_sha=commit_sha)
        valid_file_paths: set[str] = set()
        valid_file_extensions = get_all_supported_extensions()

        for file in tree.tree:
            if file.type == "blob" and any(
                file.path.endswith(ext) for ext in valid_file_extensions
            ):
                valid_file_paths.add(file.path)

        return valid_file_paths

    @functools.lru_cache(maxsize=16)
    def get_commit_history(
        self, path: str, sha: str | None = None, autocorrect: bool = False, max_commits: int = 10
    ) -> list[str]:
        if sha is None:
            sha = self.base_commit_sha

        if autocorrect:
            path, was_autocorrected = self._autocorrect_path(path, sha)
            if not was_autocorrected and path not in self.get_valid_file_paths(sha):
                return []

        commits = self.repo.get_commits(sha=sha, path=path)
        commit_list = list(commits[:max_commits])
        commit_strs = []

        def process_commit(commit):
            short_sha = commit.sha[:7]
            message = commit.commit.message
            files_touched = [
                {"path": file.filename, "status": file.status} for file in commit.files[:20]
            ]

            # Build a file tree representation instead of a flat list
            file_tree_str = self._build_file_tree_string(files_touched)

            # Add a note about additional files if needed
            additional_files_note = ""
            if len(files_touched) < len(commit.files):
                additional_files_note = (
                    f"\n[and {len(commit.files) - len(files_touched)} more files were changed...]"
                )

            string = textwrap.dedent(
                """\
                ----------------
                {short_sha} - {message}
                Files touched:
                {file_tree}{additional_files}
                """
            ).format(
                short_sha=short_sha,
                message=message,
                file_tree=file_tree_str,
                additional_files=additional_files_note,
            )
            return string

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_commit, commit_list))

        for result in results:
            commit_strs.append(result)

        return commit_strs

    def _build_file_tree_string(
        self, files: list[dict], only_immediate_children_of_path: str | None = None
    ) -> str:
        """
        Builds a tree representation of files to save tokens when many files share the same directories.
        The output is similar to the 'tree' command in terminal.

        Args:
            files: List of dictionaries with 'path' and 'status' keys
            only_immediate_children_of_path: If provided, only include files and directories that are immediate children of this path

        Returns:
            A string representation of the file tree
        """
        if not files:
            return "No files changed"

        if only_immediate_children_of_path is not None:
            only_immediate_children_of_path = only_immediate_children_of_path.rstrip("/")

        # First, build a nested dictionary structure representing the file tree
        tree: dict = {}
        for file in files:
            path = file["path"]
            status = file["status"]

            # Split the path into components
            parts = path.split("/")

            # Navigate through the tree, creating nodes as needed
            current = tree
            for i, part in enumerate(parts):
                # If this is the last part (the filename)
                if i == len(parts) - 1:
                    current[part] = {"__status__": status}
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        # Now build the tree string recursively
        lines = []

        def _build_tree(node, previous_parts=[], prefix="", is_last=True, is_root=True):
            items = list(node.items())

            # Process each item
            for i, (key, value) in enumerate(sorted(items)):
                if key == "__status__":
                    continue

                # Determine if this is the last item at this level
                is_last_item = i == len(items) - 1 or (i == len(items) - 2 and "__status__" in node)

                # Create the appropriate prefix for this line
                if is_root:
                    current_prefix = ""
                    next_prefix = ""
                else:
                    current_prefix = prefix + ("└── " if is_last else "├── ")
                    next_prefix = prefix + ("    " if is_last else "│   ")

                # If this is a file (has a status)
                if "__status__" in value:
                    if (
                        only_immediate_children_of_path is not None
                        and not only_immediate_children_of_path == ("/".join(previous_parts))
                    ):
                        continue

                    status = value["__status__"]
                    status_str = f" ({status})" if status else ""
                    lines.append(f"{current_prefix}{key}{status_str}")
                # If this is a directory
                else:
                    # If this is within the specified path
                    if (
                        only_immediate_children_of_path is None
                        or only_immediate_children_of_path.startswith(
                            "/".join(previous_parts + [key])
                        )
                    ):
                        lines.append(f"{current_prefix}{key}/")
                        _build_tree(value, previous_parts + [key], next_prefix, is_last_item, False)
                    elif (
                        only_immediate_children_of_path is not None
                        and only_immediate_children_of_path == "/".join(previous_parts)
                    ):
                        lines.append(f"{current_prefix}{key}/")

        # Start building the tree from the root
        _build_tree(tree)

        return "\n".join(lines)

    @functools.lru_cache(maxsize=16)
    def get_commit_patch_for_file(
        self, path: str, commit_sha: str, autocorrect: bool = False
    ) -> str | None:
        if autocorrect:
            path, was_autocorrected = self._autocorrect_path(path, commit_sha)
            if not was_autocorrected and path not in self.get_valid_file_paths(commit_sha):
                return None

        commit = self.repo.get_commit(commit_sha)
        matching_file = next((file for file in commit.files if file.filename == path), None)
        if not matching_file:
            return None

        return matching_file.patch

    def _create_branch(self, branch_name, from_base_sha=False):
        ref = self.repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=(
                self.base_commit_sha
                if from_base_sha
                else self.get_branch_head_sha(self.base_branch)
            ),
        )
        return ref

    @functools.lru_cache(maxsize=8)
    def get_git_tree(self, commit_sha: str) -> CompleteGitTree:
        """
        Get the git tree for a specific sha, handling truncation with divide and conquer.
        Always returns a CompleteGitTree instance for consistent interface.

        First tries to get the complete tree recursively. If truncated, it uses a
        divide and conquer approach to fetch all subtrees individually and combine them.

        Args:
            sha: The commit SHA to get the tree for

        Returns:
            A CompleteGitTree with all items from all subtrees
        """
        commit = self.repo.get_git_commit(commit_sha)
        tree = self.repo.get_git_tree(sha=commit.tree.sha, recursive=True)

        if not tree.raw_data.get("truncated", False):
            return CompleteGitTree(tree)

        complete_tree = CompleteGitTree()
        root_tree = self.repo.get_git_tree(sha=commit.tree.sha, recursive=False)

        for key, value in root_tree.raw_data.items():
            if key != "tree" and key != "truncated":
                complete_tree.raw_data[key] = value

        for item in root_tree.tree:
            complete_tree.add_item(item)

        tree_items = [item for item in root_tree.tree if item.type == "tree"]
        with ThreadPoolExecutor() as executor:
            subtree_results = []
            for item in tree_items:
                subtree_results.append(executor.submit(self._get_git_subtree, item.sha))

            for future in subtree_results:
                subtree_items = future.result()
                complete_tree.add_items(subtree_items)

        return complete_tree

    def _get_git_subtree(self, sha: str) -> list:
        """
        Process a subtree and return all its items for parallel execution.

        Args:
            sha: The SHA of the subtree

        Returns:
            A list of all tree items from this subtree and its nested subtrees
        """
        items = []
        subtree = self.repo.get_git_tree(sha=sha, recursive=True)

        if not subtree.raw_data.get("truncated", False):
            return subtree.tree

        non_recursive_subtree = self.repo.get_git_tree(sha=sha, recursive=False)

        nested_tree_items = [item for item in non_recursive_subtree.tree if item.type == "tree"]
        non_tree_items = [item for item in non_recursive_subtree.tree if item.type != "tree"]

        items.extend(non_tree_items)

        if nested_tree_items:
            with ThreadPoolExecutor() as executor:
                subtree_futures = [
                    executor.submit(self._get_git_subtree, item.sha) for item in nested_tree_items
                ]

                for future in subtree_futures:
                    items.extend(future.result())

        return items

    def process_one_file_for_git_commit(
        self, *, branch_ref: str, patch: FilePatch | None = None, change: FileChange | None = None
    ) -> InputGitTreeElement | None:
        """
        This method is used to get a single change to be committed by to github.
        It processes a FilePatch/FileChange object and converts it into an InputGitTreeElement which can be commited
        It supports both FilePatch and FileChange objects.
        """
        path = patch.path if patch else (change.path if change else None)
        patch_type = patch.type if patch else (change.change_type if change else None)
        if not path:
            raise ValueError("Path must be provided")

        if not patch_type:
            raise ValueError("Patch type must be provided")
        if patch_type == "create":
            patch_type = "A"
        elif patch_type == "delete":
            patch_type = "D"
        elif patch_type == "edit":
            patch_type = "M"

        # Remove leading slash if it exists, the github api will reject paths with leading slashes.
        if path.startswith("/"):
            path = path[1:]

        to_apply = None
        detected_encoding = "utf-8"
        if patch_type != "A":
            to_apply, detected_encoding = self.get_file_content(path, sha=branch_ref)

        new_contents = (
            patch.apply(to_apply) if patch else (change.apply(to_apply) if change else None)
        )

        # don't create a blob if the file is being deleted
        blob = self.repo.create_git_blob(new_contents, detected_encoding) if new_contents else None

        # Prevent creating tree elements with None SHA for file additions
        if patch_type == "A" and blob is None:
            return None

        # 100644 is the git code for creating a Regular non-executable file
        # https://stackoverflow.com/questions/737673/how-to-read-the-mode-field-of-git-ls-trees-output
        return InputGitTreeElement(
            path=path, mode="100644", type="blob", sha=blob.sha if blob else None
        )

    def get_branch_ref(self, branch_name: str) -> GitRef | None:
        try:
            return self.repo.get_git_ref(f"heads/{branch_name}")
        except GithubException as e:
            if e.status == 404:
                return None
            raise e

    def create_branch_from_changes(
        self,
        *,
        pr_title: str,
        file_patches: list[FilePatch] | None = None,
        file_changes: list[FileChange] | None = None,
        branch_name: str | None = None,
        from_base_sha: bool = False,
    ) -> GitRef | None:
        if not file_patches and not file_changes:
            raise ValueError("Either file_patches or file_changes must be provided")

        new_branch_name = sanitize_branch_name(branch_name or pr_title)

        try:
            branch_ref = self._create_branch(new_branch_name, from_base_sha)
        except GithubException as e:
            # only use the random suffix if the branch already exists
            if e.status == 409 or e.status == 422:
                new_branch_name = f"{new_branch_name}-{generate_random_string(n=6)}"
                branch_ref = self._create_branch(new_branch_name, from_base_sha)
            else:
                raise e

        tree_elements = []
        if file_patches:
            for patch in file_patches:
                try:
                    element = self.process_one_file_for_git_commit(
                        branch_ref=branch_ref.ref, patch=patch
                    )
                    if element:
                        tree_elements.append(element)
                except Exception as e:
                    logger.exception(f"Error processing file patch: {e}")

        elif file_changes:
            for change in file_changes:
                try:
                    element = self.process_one_file_for_git_commit(
                        branch_ref=branch_ref.ref, change=change
                    )
                    if element:
                        tree_elements.append(element)
                except Exception as e:
                    logger.exception(f"Error processing file change: {e}")
        # latest commit is the head of new branch
        latest_commit = self.repo.get_git_commit(self.get_branch_head_sha(new_branch_name))
        base_tree = latest_commit.tree
        new_tree = self.repo.create_git_tree(tree_elements, base_tree)

        new_commit = self.repo.create_git_commit(
            message=pr_title, tree=new_tree, parents=[latest_commit]
        )

        branch_ref.edit(sha=new_commit.sha)

        # Check that the changes were made
        comparison = self.repo.compare(
            self.get_branch_head_sha(self.base_branch), branch_ref.object.sha
        )

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
    ) -> PullRequest:
        pulls = self.repo.get_pulls(state="open", head=f"{self.repo_owner}:{branch.ref}")

        if pulls.totalCount > 0:
            logger.error(
                f"Branch {branch.ref} already has an open PR.",
                extra={
                    "branch_ref": branch.ref,
                    "title": title,
                    "description": description,
                    "provided_base": provided_base,
                },
            )

            return pulls[0]

        try:
            return self.repo.create_pull(
                title=title,
                body=description,
                base=provided_base or self.base_branch or self.get_default_branch(),
                head=branch.ref,
                draft=True,
            )
        except GithubException as e:
            if e.status == 422 and "Draft pull requests are not supported" in str(e):
                # fallback to creating a regular PR if draft PR is not supported
                return self.repo.create_pull(
                    title=title,
                    body=description,
                    base=provided_base or self.base_branch or self.get_default_branch(),
                    head=branch.ref,
                    draft=False,
                )
            else:
                logger.exception("Error creating PR")
                raise e

    def get_index_file_set(
        self,
        commit_sha: str | None = None,
        max_file_size_bytes=2 * 1024 * 1024,
        skip_empty_files=False,
    ) -> set[str]:
        if commit_sha is None:
            commit_sha = self.base_commit_sha

        tree = self.get_git_tree(commit_sha=commit_sha)
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

    def post_unit_test_reference_to_original_pr_codecov_app(
        self, original_pr_url: str, unit_test_pr_url: str
    ):
        original_pr_id = int(original_pr_url.split("/")[-1])
        repo_name = original_pr_url.split("github.com/")[1].split("/pull")[0]
        url = f"https://api.github.com/repos/{repo_name}/issues/{original_pr_id}/comments"
        comment = f"Codecov has generated a new [PR]({unit_test_pr_url}) with unit tests for this PR. View the new PR({unit_test_pr_url}) to review the changes."
        params = {"body": comment}
        headers = self._get_auth_headers()
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()
        return response.json()["html_url"]

    def post_unit_test_not_generated_message_to_original_pr(self, original_pr_url: str):
        original_pr_id = int(original_pr_url.split("/")[-1])
        repo_name = original_pr_url.split("github.com/")[1].split("/pull")[0]
        url = f"https://api.github.com/repos/{repo_name}/issues/{original_pr_id}/comments"
        comment = "Sentry has determined that unit tests are not necessary for this PR."
        params = {"body": comment}
        headers = self._get_auth_headers()
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()
        return response.json()["html_url"]

    def post_issue_comment(self, pr_url: str, comment: str):
        """
        Create an issue comment on a GitHub issue (all pull requests are issues).
        This can be used to create an overall PR comment instead of associated with a specific line.
        See https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#create-an-issue-comment
        Note that expected input is pr_url NOT pr_html_url
        """
        pr_id = self._extract_id_from_pr_url(pr_url)
        issue = self.repo.get_issue(number=pr_id)
        comment_obj = issue.create_comment(body=comment)
        return comment_obj.html_url

    def post_pr_review_comment(self, pr_url: str, comment: GithubPrReviewComment):
        """
        Create a review comment on a GitHub pull request.
        See https://docs.github.com/en/rest/pulls/comments?apiVersion=2022-11-28#create-a-review-comment-for-a-pull-request
        Note that expected input is pr_url NOT pr_html_url
        """
        pr_id = self._extract_id_from_pr_url(pr_url)
        pr = self.repo.get_pull(number=pr_id)
        commit = self.repo.get_commit(comment["commit_id"])

        review_comment = pr.create_review_comment(
            body=comment["body"],
            commit=commit,
            path=comment["path"],
            line=comment.get("line", GithubObject.NotSet),
            side=comment.get("side", GithubObject.NotSet),
            start_line=comment.get("start_line", GithubObject.NotSet),
        )
        return review_comment.html_url

    def push_new_commit_to_pr(
        self,
        pr,
        commit_message: str,
        file_patches: list[FilePatch] | None = None,
        file_changes: list[FileChange] | None = None,
    ):
        if not file_patches and not file_changes:
            raise ValueError("Must provide file_patches or file_changes")
        branch_name = pr.head.ref
        tree_elements = []
        if file_patches:
            for patch in file_patches:
                element = self.process_one_file_for_git_commit(branch_ref=branch_name, patch=patch)
                if element:
                    tree_elements.append(element)
        elif file_changes:
            for change in file_changes:
                element = self.process_one_file_for_git_commit(
                    branch_ref=branch_name, change=change
                )
                if element:
                    tree_elements.append(element)
        if not tree_elements:
            logger.warning("No valid changes to commit")
            return None
        latest_sha = self.get_branch_head_sha(branch_name)
        latest_commit = self.repo.get_git_commit(latest_sha)
        base_tree = latest_commit.tree
        new_tree = self.repo.create_git_tree(tree_elements, base_tree)
        new_commit = self.repo.create_git_commit(
            message=commit_message, tree=new_tree, parents=[latest_commit]
        )
        branch_ref = self.repo.get_git_ref(f"heads/{branch_name}")
        branch_ref.edit(sha=new_commit.sha)
        return new_commit
