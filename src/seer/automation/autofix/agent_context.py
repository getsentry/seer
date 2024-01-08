import logging
import os
import shutil
import tarfile
import tempfile
import time
from typing import List

import joblib
import requests
import torch
from github import Auth, Github, GithubIntegration
from github.GitRef import GitRef
from github.Repository import Repository
from llama_index import ServiceContext
from llama_index.indices import VectorStoreIndex
from llama_index.node_parser import CodeSplitter
from llama_index.readers import SimpleDirectoryReader
from llama_index.schema import BaseNode, Document

from .types import AutofixAgentsOutput, FileChange, IssueDetails
from .utils import SentenceTransformersEmbedding

logger = logging.getLogger("autofix")


class AgentContext:
    model: str

    github: Github
    repo: Repository
    base_sha: str

    tmp_dir: str
    tmp_repo_path: str

    embed_model: SentenceTransformersEmbedding
    persist_path: str | None

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        model: str,
        github_auth: Auth.AppAuth,
        gpu_device: torch.device,
        persist_path: str | None = None,
        ref: str | None = None,
        base_sha: str | None = None,
        tmp_dir: str | None = None,
    ):
        self.model = model

        gi = GithubIntegration(auth=github_auth)
        installation = gi.get_repo_installation(repo_owner, repo_name)
        app_auth = github_auth.get_installation_auth(installation.id)

        self.github = Github(auth=app_auth)
        self.repo = self.github.get_repo(repo_owner + "/" + repo_name)

        if base_sha:
            self.base_sha = base_sha
        else:
            if ref is None:
                raise ValueError("ref cannot be None if base_sha is not provided")
            self.base_sha = self.repo.get_git_ref(ref).object.sha

        if tmp_dir is None:
            tmp_dir = os.path.join(
                tempfile.gettempdir(), f"{repo_owner}-{repo_name}_{self.base_sha}"
            )
        self.tmp_dir = tmp_dir
        self.tmp_repo_path = os.path.join(self.tmp_dir, f"repo")

        logger.info(f"Using tmp dir {self.tmp_dir}")

        self.embed_model = SentenceTransformersEmbedding("thenlper/gte-small", device=gpu_device)
        self.persist_path = persist_path

    def _embed_and_index_nodes(self, nodes: list[BaseNode]) -> VectorStoreIndex:
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
        logger.debug(f"Embedding {len(nodes)} nodes")
        index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)

        return index

    def _documents_to_nodes(self, documents: list[Document], max_file_size_bytes=1024 * 1024 * 2):
        file_extension_mapping = {
            # "ts": "typescript",
            # "tsx": "tsx",
            "py": "python"
        }
        documents_by_language = {}

        for document in documents:
            file_extension = document.metadata["file_name"].split(".")[-1]
            language = file_extension_mapping.get(file_extension, "unknown")

            if document.metadata["file_size"] > max_file_size_bytes:
                continue

            if language != "unknown":
                document.metadata["file_path"] = document.metadata["file_path"].replace(
                    self.tmp_repo_path.rstrip("/") + "/", "", 1
                )
                documents_by_language.setdefault(language, []).append(document)

        print(f"example filepath! {documents_by_language['python'][0].metadata['file_path'] }")
        nodes: List[BaseNode] = []

        for language, language_docs in documents_by_language.items():
            logger.debug(f"{language}: {len(language_docs)}")

            splitter = CodeSplitter(
                language=language,
                chunk_lines=16,  # lines per chunk
                chunk_lines_overlap=4,  # lines overlap between chunks
                max_chars=5096,  # max chars per chunk
            )

            nodes.extend(splitter.get_nodes_from_documents(language_docs, show_progress=False))

        return nodes

    def _load_repo_to_tmp_dir(self):
        # Check if output directory exists, if not create it
        os.makedirs(self.tmp_repo_path, exist_ok=True)
        for root, dirs, files in os.walk(self.tmp_repo_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        tarball_url = self.repo.get_archive_link("tarball", ref=self.base_sha)

        response = requests.get(tarball_url, stream=True)
        if response.status_code == 200:
            with open(f"{self.tmp_dir}/repo-{self.base_sha}.tar.gz", "wb") as f:
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
        with tarfile.open(f"{self.tmp_dir}/repo-{self.base_sha}.tar.gz", "r:gz") as tar:
            tar.extractall(path=self.tmp_repo_path)  # extract all members normally
            extracted_folders = [
                name
                for name in os.listdir(self.tmp_repo_path)
                if os.path.isdir(os.path.join(self.tmp_repo_path, name))
            ]
            if extracted_folders:
                root_folder = extracted_folders[0]  # assuming the first folder is the root folder
                root_folder_path = os.path.join(self.tmp_repo_path, root_folder)
                for item in os.listdir(root_folder_path):
                    s = os.path.join(root_folder_path, item)
                    d = os.path.join(self.tmp_repo_path, item)
                    if os.path.isdir(s):
                        shutil.move(
                            s, d
                        )  # move all directories from the root folder to the output directory
                    else:
                        shutil.copy2(
                            s, d
                        )  # copy all files from the root folder to the output directory
                shutil.rmtree(root_folder_path)  # remove the root folder

        # Delete the tar file
        try:
            os.remove(f"{self.tmp_dir}/repo-{self.base_sha}.tar.gz")
        except OSError as e:
            logger.error(f"Failed to delete tar file: {e}")

    def _load_data_from_github(self):
        self._load_repo_to_tmp_dir()
        documents = SimpleDirectoryReader(
            self.tmp_repo_path, required_exts=[".py"], recursive=True
        ).load_data()
        nodes = self._documents_to_nodes(documents)

        self.documents = documents
        self.nodes = nodes

        return documents, nodes

    def get_data(self, cache_only=False):
        documents, nodes = self._load_data_from_github()

        persist_path = os.path.join(self.persist_path, self.base_sha) if self.persist_path else None
        if persist_path and os.path.exists(persist_path):
            logger.debug(f"Loading index from {persist_path}")
            index: VectorStoreIndex = joblib.load(persist_path)
            index._service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
        else:
            if cache_only:
                raise Exception(f"Could not load index from {persist_path}")
            logger.debug("Creating index")
            index = self._embed_and_index_nodes(nodes)

            if persist_path:
                os.makedirs(os.path.dirname(persist_path), exist_ok=True)
                joblib.dump(index, persist_path)
                logger.debug(f"Saved index to {persist_path}")

        return index, documents, nodes

    def cleanup(self):
        shutil.rmtree(self.tmp_dir)
        logger.info(f"Deleted tmp dir {self.tmp_dir}")

    def _create_branch(self, branch_name, base_branch=None, base_commit_sha: str | None = None):
        base_sha = base_commit_sha
        if base_branch:
            base_sha = self.repo.get_branch(base_branch).commit.sha

        if not base_sha:
            raise ValueError("base_sha cannot be None")

        ref = self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
        return ref

    def commit_file_change(self, path, message, content, branch):
        contents = self.repo.get_contents(path, ref=branch)

        if isinstance(contents, list):
            raise Exception(f"Expected a single ContentFile but got a list for path {path}")

        self.repo.update_file(path, message, content, contents.sha, branch=branch)

    def get_file_contents(self, path, ref):
        logger.debug(f"Getting file contents for {path} in {self.repo.name} on ref {ref}")
        try:
            contents = self.repo.get_contents(path, ref=ref)

            if isinstance(contents, list):
                raise Exception(f"Expected a single ContentFile but got a list for path {path}")

            return contents.decoded_content.decode()
        except Exception as e:
            logger.error(f"Error getting file contents: {e}")

            return f"Error: file with path {path} not found in {self.repo.name} on ref {ref}"

    def create_branch_from_changes(self, file_changes: List[FileChange], base_commit_sha) -> GitRef:
        new_branch_name = f"test-branch-{time.time()}"
        branch_ref = self._create_branch(new_branch_name, base_commit_sha=base_commit_sha)

        for change in file_changes:
            try:
                self.commit_file_change(
                    change.path, change.description, change.contents, new_branch_name
                )
            except Exception as e:
                logger.error(f"Error committing file change: {e}")

        return branch_ref

    def _get_stats_str(self, prompt_tokens: int, completion_tokens: int):
        gpt4turbo_prompt_price_per_token = 0.01 / 1000
        gpt4turbo_completion_price_per_token = 0.03 / 1000

        prompt_price = prompt_tokens * gpt4turbo_prompt_price_per_token
        completion_price = completion_tokens * gpt4turbo_completion_price_per_token
        total_price = prompt_price + completion_price

        stats_str = f"""Model: {self.model}
Prompt tokens: **{prompt_tokens} (${prompt_price:.3f})**
Completion tokens: **{completion_tokens} (${completion_price:.3f})**
Total tokens: **{prompt_tokens + completion_tokens} (${total_price:.3f})**"""

        return stats_str

    def create_pr_from_branch(
        self, branch: GitRef, autofix_output: AutofixAgentsOutput, issue_id: str
    ):
        title = f"""🤖 {autofix_output.title}"""

        issue_link = f"https://sentry.io/organizations/sentry/issues/{issue_id}/"

        description = f"""👋 Hi there! This PR was automatically generated 🤖

{autofix_output.description}

### Issue that triggered this PR:
{issue_link}

### Stats:
{self._get_stats_str(autofix_output.usage.prompt_tokens, autofix_output.usage.completion_tokens)}"""

        return self.repo.create_pull(
            title=title,
            body=description,
            base="master",
            head=branch.ref,
        )
