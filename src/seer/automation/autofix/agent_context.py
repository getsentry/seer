import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from typing import List

import joblib
import requests
import sentry_sdk
import torch
from github import Auth, Github, GithubIntegration
from github.GitRef import GitRef
from github.Repository import Repository
from llama_index import ServiceContext
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices import VectorStoreIndex
from llama_index.node_parser import CodeSplitter
from llama_index.readers import SimpleDirectoryReader
from llama_index.schema import BaseNode, Document
from llama_index.storage import StorageContext
from unidiff import PatchSet

from seer.automation.autofix.types import AutofixAgentsOutput, FileChange
from seer.automation.autofix.utils import (
    MemoryVectorStore,
    SentenceTransformersEmbedding,
    generate_random_string,
    sanitize_branch_name,
)

logger = logging.getLogger("autofix")

CACHED_COMMIT_SHA = "ac63e3750291a6795d31404030783358ef3ea1ac"


class AgentContext:
    model: str

    github: Github
    repo: Repository
    base_sha: str
    bucket = "sentry-ml"

    tmp_dir: str
    tmp_repo_path: str

    embed_model: SentenceTransformersEmbedding

    index: VectorStoreIndex
    documents: list[Document]
    nodes: list[BaseNode]

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        model: str,
        github_auth: Auth.AppAuth,
        gpu_device: torch.device,
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
        self.cached_commit_json_path = os.path.join("./", "models/autofix_cached_commit.json")

        logger.info(f"Using tmp dir {self.tmp_dir}")

        self.embed_model = SentenceTransformersEmbedding("thenlper/gte-small", device=gpu_device)

        self._get_data()

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
        documents_by_language: dict[str, list[Document]] = {}

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

        nodes: List[BaseNode] = []

        for language, language_docs in documents_by_language.items():
            logger.debug(f"{language}: {len(language_docs)}")

            splitter = CodeSplitter(
                language=language,
                chunk_lines=16,  # lines per chunk
                chunk_lines_overlap=4,  # lines overlap between chunks
                max_chars=2048,  # max chars per chunk
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
        logger.debug(f"Loading data from github for {self.repo.name} on ref {self.base_sha}")
        self._load_repo_to_tmp_dir()
        documents = SimpleDirectoryReader(
            self.tmp_repo_path, required_exts=[".py"], recursive=True
        ).load_data()
        nodes = self._documents_to_nodes(documents)

        return documents, nodes

    def _get_commit_file_diffs(self) -> tuple[list[str], list[str]]:
        with open(self.cached_commit_json_path, "r") as file:
            cached_commit_data = json.load(file)
        cached_commit_sha = cached_commit_data.get("sha")

        comparison = self.repo.compare(cached_commit_sha, self.base_sha)

        requester = self.repo._requester

        # Hack: We're extracting the authorization and user agent headers from the PyGithub library to get this diff
        # This has to be done because the files list inside the comparison object is limited to only 300 files.
        # We get the entire diff from the diff object returned from the `diff_url`
        headers = {
            "Authorization": f"{requester._Requester__auth.token_type} {requester._Requester__auth.token}",  # type: ignore
            "User-Agent": requester._Requester__userAgent,  # type: ignore
        }
        data = requests.get(comparison.diff_url, headers=headers).content

        patch_set = PatchSet(data.decode("utf-8"))

        added_files = [patch.path for patch in patch_set.added_files]
        modified_files = [patch.path for patch in patch_set.modified_files]
        removed_files = [patch.path for patch in patch_set.removed_files]

        changed_files = list(set(added_files + modified_files))

        return changed_files, removed_files

    def _get_data(self):
        documents, nodes = self._load_data_from_github()

        logger.debug(f"Loading index from storage context")

        storage_path = os.path.join("./", "models/autofix_storage_context/")

        service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
        memory_vector_store = MemoryVectorStore().from_persist_dir(storage_path)
        storage_context = StorageContext.from_defaults(
            vector_store=memory_vector_store,
            persist_dir=storage_path,
        )
        index_structs = storage_context.index_store.index_structs()

        if len(index_structs) == 0:
            raise Exception("No index structures found in storage context")
        index_struct: IndexDict = index_structs[0]  # type: ignore

        index = VectorStoreIndex(
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True,
        )

        logger.debug(f"Loaded index from storage context '{storage_path}'.")

        # Update the documents that changed in the diff.
        changed_files, deleted_files = self._get_commit_file_diffs()

        logger.debug(
            f"Updating index with {len(changed_files)} changed files and {len(deleted_files)} deleted files"
        )

        self.index = index
        self.documents = documents
        self.nodes = nodes

        documents_to_update = [
            document for document in documents if document.metadata["file_path"] in changed_files
        ]
        documents_to_delete = [
            document for document in documents if document.metadata["file_path"] in deleted_files
        ]

        for document in documents_to_update + documents_to_delete:
            self.index.delete(document.get_doc_id())

        with sentry_sdk.start_span(
            op="seer.automation.autofix.indexing",
            description="Indexing the diff between the cached commit and the requested commit",
        ) as span:
            new_nodes = self._documents_to_nodes(documents_to_update)
            self.index.insert_nodes(new_nodes)

            span.set_tag("num_documents", len(documents_to_update))
            span.set_tag("num_nodes", len(new_nodes))

        storage_context.persist(persist_dir=storage_path)
        memory_vector_store.persist(persist_path=os.path.join(storage_path, "vector_store.json"))

        try:
            os.remove(os.path.join(storage_path, "default__vector_store.json"))
        except OSError as e:
            logger.error(f"Failed to delete default vector store file: {e}")

        with open(self.cached_commit_json_path, "w") as sha_file:
            json.dump({"sha": self.base_sha}, sha_file)

    def _get_document(self, file_path: str):
        for document in self.documents:
            if file_path in document.metadata["file_path"]:
                return document

        return None

    def _update_document(self, file_path: str, contents: str | None):
        document = self._get_document(file_path)
        if document:
            # Delete operation
            if contents is None:
                self.index.delete(document.get_doc_id())
                self.documents.remove(document)
                # Also remove associated nodes
                associated_nodes = [
                    node for node in self.nodes if node.ref_doc_id == document.get_doc_id()
                ]
                for node in associated_nodes:
                    self.nodes.remove(node)
            # Update operation
            else:
                self.index.delete(document.get_doc_id())
                new_doc = Document(text=contents)
                new_doc.metadata = document.metadata
                new_nodes = self._documents_to_nodes([new_doc])
                self.index.insert_nodes(new_nodes)
        # Create operation
        else:
            if contents is not None:
                document = Document(text=contents)
                document.metadata = {"file_path": file_path, "file_name": file_path.split("/")[-1]}
                new_nodes = self._documents_to_nodes([document])
                self.index.insert_nodes(new_nodes)
                self.documents.append(document)
                self.nodes.extend(new_nodes)

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

    def create_branch_from_changes(
        self, pr_title: str, file_changes: List[FileChange], base_commit_sha
    ) -> GitRef:
        new_branch_name = f"autofix/{sanitize_branch_name(pr_title)}/{generate_random_string(n=6)}"
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
        self, branch: GitRef, autofix_output: AutofixAgentsOutput, issue_id: int | str
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
