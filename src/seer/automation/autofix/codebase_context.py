import json
import logging
import os
import shutil
import tarfile
import tempfile
from typing import List

import requests
import sentry_sdk
from filelock import SoftFileLock, Timeout
from llama_index import ServiceContext
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices import VectorStoreIndex
from llama_index.node_parser import CodeSplitter
from llama_index.readers import SimpleDirectoryReader
from llama_index.schema import BaseNode, Document
from llama_index.storage import StorageContext

from seer.automation.autofix.repo_client import RepoClient
from seer.automation.autofix.utils import (
    MemoryVectorStore,
    SentenceTransformersEmbedding,
    get_torch_device,
)

logger = logging.getLogger("autofix")

CACHED_COMMIT_JSON_PATH = os.path.join("./", "models/autofix_cached_commit.json")
STORAGE_DIR = os.path.join("./", "models/autofix_storage_context/")
LOCK_PATH = os.path.join(STORAGE_DIR, "lock")


class CodebaseContext:
    repo_client: RepoClient
    base_sha: str

    tmp_dir: str
    tmp_repo_path: str

    embed_model: SentenceTransformersEmbedding

    index: VectorStoreIndex
    documents: list[Document]
    nodes: list[BaseNode]

    def __init__(self, repo_client: RepoClient, base_sha: str):
        self.repo_client = repo_client
        self.base_sha = base_sha

        self.tmp_dir = tempfile.mkdtemp(
            prefix=f"{repo_client.repo_owner}-{repo_client.repo_name}_{self.base_sha}"
        )
        self.tmp_repo_path = os.path.join(self.tmp_dir, f"repo")

        logger.info(f"Using tmp dir {self.tmp_dir}")

        self.embed_model = SentenceTransformersEmbedding(
            "thenlper/gte-small", device=get_torch_device()
        )

        self._get_data()

    @staticmethod
    def get_cached_commit_sha() -> str | None:
        try:
            with open(CACHED_COMMIT_JSON_PATH, "r") as file:
                cached_commit_data = json.load(file)
                cached_commit_sha = cached_commit_data.get("sha")
                return cached_commit_sha
        except IOError:
            logger.error(f"Failed to read cached commit file: {CACHED_COMMIT_JSON_PATH}")
            return None

    @staticmethod
    def set_cached_commit_sha(sha: str):
        try:
            with open(CACHED_COMMIT_JSON_PATH, "w") as sha_file:
                json.dump({"sha": sha}, sha_file)
        except IOError:
            logger.error(f"Failed to write cached commit file: {CACHED_COMMIT_JSON_PATH}")

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

        tarball_url = self.repo_client.repo.get_archive_link("tarball", ref=self.base_sha)

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
        logger.debug(
            f"Loading data from github for {self.repo_client.repo.full_name} on ref {self.base_sha}"
        )
        self._load_repo_to_tmp_dir()
        documents = SimpleDirectoryReader(
            self.tmp_repo_path, required_exts=[".py"], recursive=True
        ).load_data()
        nodes = self._documents_to_nodes(documents)

        return documents, nodes

    def _get_data(self):
        # The files will be locked for the entire process of loading data; that means only one worker at a time can go through the data loading process which is a bottleneck.
        # However, this should be a temporary compromise during PoC stage to ensure that only the latest commit's embeddings are stored.
        try:
            documents, nodes = self._load_data_from_github()

            logger.debug(f"Loading index from storage context")

            with SoftFileLock(
                os.path.join(LOCK_PATH), timeout=60 * 60
            ):  # 1 hour max read lock timeout
                logger.debug(f"Acquired lock for {STORAGE_DIR}")
                with sentry_sdk.start_span(
                    op="seer.automation.autofix.index_loading",
                    description="Loading the vector store index from local filesystem",
                ) as span:
                    cached_commit_sha = self.get_cached_commit_sha()

                    assert cached_commit_sha is not None, "Cached commit SHA not found"

                    service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
                    memory_vector_store = MemoryVectorStore().from_persist_dir(STORAGE_DIR)
                    storage_context = StorageContext.from_defaults(
                        vector_store=memory_vector_store,
                        persist_dir=STORAGE_DIR,
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

                self.index = index
                self.documents = documents
                self.nodes = nodes

                logger.debug(f"Loaded index from storage context '{STORAGE_DIR}'.")

            logger.debug(f"Released lock for {STORAGE_DIR}")

        finally:
            # Always cleanup the tmp dir
            self.cleanup()

    def update_codebase_index(self):
        cached_commit_sha = self.get_cached_commit_sha()

        assert cached_commit_sha is not None, "Cached commit SHA not found"

        changed_files, deleted_files = self.repo_client.get_commit_file_diffs(
            cached_commit_sha, self.base_sha
        )

        logger.debug(
            f"Updating index with {len(changed_files)} changed files and {len(deleted_files)} deleted files"
        )

        documents_to_update = [
            document
            for document in self.documents
            if document.metadata["file_path"] in changed_files
        ]
        documents_to_delete = [
            document
            for document in self.documents
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

        commit_is_newer = (
            self.repo_client.repo.compare(cached_commit_sha, self.base_sha).ahead_by > 0
        )

        if commit_is_newer:
            # Save only when the commit is newer
            logger.debug(f"Saving index to storage context")
            try:
                with SoftFileLock(
                    os.path.join(LOCK_PATH), timeout=60 * 15
                ):  # 15 minute max write lock timeout
                    logger.debug(f"Acquired lock for {STORAGE_DIR}")
                    self.index.storage_context.persist(persist_dir=STORAGE_DIR)
                    self.index.vector_store.persist(
                        persist_path=os.path.join(STORAGE_DIR, "vector_store.json")
                    )

                    try:
                        os.remove(os.path.join(STORAGE_DIR, "default__vector_store.json"))
                    except OSError as e:
                        logger.error(f"Failed to delete default vector store file: {e}")

                    self.set_cached_commit_sha(self.base_sha)
                logger.debug(f"Released lock for {STORAGE_DIR}")
            except Timeout:
                logger.warning(
                    f"Timed out waiting for lock to save {STORAGE_DIR}, skipping save..."
                )

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
        logger.debug(
            f"Getting file contents for {path} in {self.repo_client.repo.full_name} on ref {ref}"
        )
        try:
            contents = self.repo_client.repo.get_contents(path, ref=ref)

            if isinstance(contents, list):
                raise Exception(f"Expected a single ContentFile but got a list for path {path}")

            return contents.decoded_content.decode()
        except Exception as e:
            logger.error(f"Error getting file contents: {e}")

            return f"Error: file with path {path} not found in {self.repo_client.repo.full_name} on ref {ref}"

    def cleanup(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            logger.info(f"Deleted tmp dir {self.tmp_dir}")
        else:
            logger.info(f"Tmp dir {self.tmp_dir} already removed")
