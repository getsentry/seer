import json
import logging
import os
import shutil
import tarfile
import tempfile
from typing import List

import requests
import sentry_sdk
import torch
from filelock import SoftFileLock
from github import Auth, Github, GithubIntegration
from github.Repository import Repository
from llama_index import ServiceContext
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices import VectorStoreIndex
from llama_index.node_parser import CodeSplitter
from llama_index.readers import SimpleDirectoryReader
from llama_index.schema import BaseNode, Document
from llama_index.storage import StorageContext
from unidiff import PatchSet

from seer.automation.agent.types import Usage
from seer.automation.autofix.repo_client import get_github_auth
from seer.automation.autofix.utils import (
    MemoryVectorStore,
    SentenceTransformersEmbedding,
    get_torch_device,
)

logger = logging.getLogger("autofix")


class CodebaseContext:
    github_auth: Auth.Token | Auth.AppInstallationAuth
    github: Github
    repo: Repository
    base_sha: str

    tmp_dir: str
    tmp_repo_path: str

    embed_model: SentenceTransformersEmbedding

    index: VectorStoreIndex
    documents: list[Document]
    nodes: list[BaseNode]

    def __init__(self, repo_owner: str, repo_name: str, base_sha: str):
        self.github = Github(auth=get_github_auth(repo_owner, repo_name))
        self.repo = self.github.get_repo(repo_owner + "/" + repo_name)

        self.base_sha = base_sha

        self.tmp_dir = tempfile.mkdtemp(prefix=f"{repo_owner}-{repo_name}_{self.base_sha}")
        self.tmp_repo_path = os.path.join(self.tmp_dir, f"repo")
        self.cached_commit_json_path = os.path.join("./", "models/autofix_cached_commit.json")

        logger.info(f"Using tmp dir {self.tmp_dir}")

        self.embed_model = SentenceTransformersEmbedding(
            "thenlper/gte-small", device=get_torch_device()
        )

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
            splitter = CodeSplitter(
                language=language,
                chunk_lines=16,  # lines per chunk
                chunk_lines_overlap=4,  # lines overlap between chunks
                max_chars=2048,  # max chars per chunk
            )

            language_nodes = splitter.get_nodes_from_documents(language_docs, show_progress=False)

            logger.debug(
                f"Docs/nodes for {language}: {len(language_docs)} documents split into {len(language_nodes)} nodes"
            )

            nodes.extend(language_nodes)

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

    def _get_commit_file_diffs(self, prev_sha: str) -> tuple[list[str], list[str]]:
        """
        Returns the list of files to change and files to delete in the diff in order to turn a commit into another.
        """
        comparison = self.repo.compare(prev_sha, self.base_sha)

        # Support reverse diffs, because the api would return an empty list of files if the comparison is behind
        is_behind = comparison.status == "behind"
        if is_behind:
            comparison = self.repo.compare(self.base_sha, prev_sha)

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

    def _get_data(self):
        # The files will be locked for the entire process of loading data; that means only one worker at a time can go through the data loading process which is a bottleneck.
        # However, this should be a temporary compromise during PoC stage to ensure that only the latest commit's embeddings are stored.
        try:
            documents, nodes = self._load_data_from_github()

            logger.debug(f"Loading index from storage context")

            storage_dir = os.path.join("./", "models/autofix_storage_context/")
            lock_path = os.path.join(storage_dir, "lock")

            with SoftFileLock(os.path.join(lock_path), timeout=60 * 60):  # 1 hour max lock timeout
                logger.debug(f"Acquired lock for {storage_dir}")
                with sentry_sdk.start_span(
                    op="seer.automation.autofix.index_loading",
                    description="Loading the vector store index from local filesystem",
                ) as span:
                    with open(self.cached_commit_json_path, "r") as file:
                        cached_commit_data = json.load(file)
                    cached_commit_sha = cached_commit_data.get("sha")

                    service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
                    memory_vector_store = MemoryVectorStore().from_persist_dir(storage_dir)
                    storage_context = StorageContext.from_defaults(
                        vector_store=memory_vector_store,
                        persist_dir=storage_dir,
                    )
                    index_structs = storage_context.index_store.index_structs()

                    if len(index_structs) == 0:
                        raise Exception("No index structures found in storage context")
                    index_struct: IndexDict = index_structs[0]  # type: ignore

                    index = VectorStoreIndex(
                        index_struct=index_struct,
                        service_context=service_context,
                        storage_context=storage_context,
                        show_progress=False,
                    )

                logger.debug(f"Loaded index from storage context '{storage_dir}'.")

                # Update the documents that changed in the diff.
                changed_files, deleted_files = self._get_commit_file_diffs(cached_commit_sha)

                logger.debug(
                    f"Updating index with {len(changed_files)} changed files and {len(deleted_files)} deleted files"
                )

                self.index = index
                self.documents = documents
                self.nodes = nodes

                documents_to_update = [
                    document
                    for document in documents
                    if document.metadata["file_path"] in changed_files
                ]
                documents_to_delete = [
                    document
                    for document in documents
                    if document.metadata["file_path"] in deleted_files
                ]

                for document in documents_to_update + documents_to_delete:
                    self.index.delete(document.get_doc_id())

                with sentry_sdk.start_span(
                    op="seer.automation.autofix.indexing",
                    description="Indexing the diff between the cached commit and the requested commit",
                ) as span:
                    logger.debug(f"Begin index update...")
                    new_nodes = self._documents_to_nodes(documents_to_update)
                    self.index.insert_nodes(new_nodes)

                    span.set_tag("num_documents", len(documents_to_update))
                    span.set_tag("num_nodes", len(new_nodes))

                commit_is_newer = self.repo.compare(cached_commit_sha, self.base_sha).ahead_by > 0

                if commit_is_newer:
                    # Save only when the commit is newer
                    storage_context.persist(persist_dir=storage_dir)
                    memory_vector_store.persist(
                        persist_path=os.path.join(storage_dir, "vector_store.json")
                    )

                    try:
                        os.remove(os.path.join(storage_dir, "default__vector_store.json"))
                    except OSError as e:
                        logger.error(f"Failed to delete default vector store file: {e}")

                    with open(self.cached_commit_json_path, "w") as sha_file:
                        json.dump({"sha": self.base_sha}, sha_file)

                logger.debug(f"Released lock for {storage_dir}")

        finally:
            # Always cleanup the tmp dir
            self.cleanup()

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
                self.index.delete_ref_doc(document.get_doc_id())
                self.documents.remove(document)
                # Also remove associated nodes
                associated_nodes = [
                    node for node in self.nodes if node.ref_doc_id == document.get_doc_id()
                ]
                for node in associated_nodes:
                    self.nodes.remove(node)
            # Update operation
            else:
                self.index.delete_ref_doc(document.get_doc_id())
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

    def cleanup(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            logger.info(f"Deleted tmp dir {self.tmp_dir}")
        else:
            logger.info(f"Tmp dir {self.tmp_dir} already removed")
