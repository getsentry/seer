import difflib
import logging
from typing import Type

import numpy as np
import sentry_sdk
import torch
import tree_sitter_languages
from sentence_transformers import SentenceTransformer
from sentry_sdk.ai.monitoring import ai_track
from tree_sitter import Tree
from unidiff import PatchSet

from seer.automation.codebase.ast import (
    extract_declaration,
    find_first_parent_declaration,
    supports_parent_declarations,
)
from seer.automation.codebase.models import (
    BaseDocumentChunk,
    ChunkQueryResult,
    CodebaseNamespace,
    CodebaseNamespaceStatus,
    Document,
    DraftDocument,
    EmbeddedDocumentChunk,
    QueryResultDocumentChunk,
    RepositoryInfo,
)
from seer.automation.codebase.namespace import CodebaseNamespaceManager
from seer.automation.codebase.parser import DocumentParser
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.state import CodebaseStateManager, DummyCodebaseStateManager
from seer.automation.codebase.utils import (
    cleanup_dir,
    get_language_from_path,
    group_documents_by_language,
    potential_frame_match,
    read_specific_files,
)
from seer.automation.models import (
    EventDetails,
    FileChange,
    FilePatch,
    Hunk,
    InitializationError,
    Line,
    RepoDefinition,
    Stacktrace,
)
from seer.automation.state import State
from seer.db import DbRepositoryInfo, Session
from seer.utils import class_method_lru_cache

logger = logging.getLogger("autofix")


class CodebaseIndex:
    def __init__(
        self,
        organization: int,
        project: int,
        repo_client: RepoClient,
        workspace: CodebaseNamespaceManager,
        state_manager: CodebaseStateManager,
        embedding_model: SentenceTransformer,
    ):
        self.repo_client = repo_client
        self.organization = organization
        self.project = project
        self.embedding_model = embedding_model
        self.workspace = workspace
        self.state_manager = state_manager

        logger.info(
            f"Loaded codebase index for {repo_client.repo.full_name}, {'with existing data' if self.repo_info else 'without existing data'}"
        )

    @property
    def repo_info(self) -> RepositoryInfo:
        return self.workspace.repo_info

    @property
    def namespace(self) -> CodebaseNamespace:
        return self.workspace.namespace

    @staticmethod
    def get_repo_info_from_db(repo_id: int):
        db_repo_info = Session().get(DbRepositoryInfo, repo_id)

        return RepositoryInfo.from_db(db_repo_info) if db_repo_info else None

    @staticmethod
    def has_repo_been_indexed(
        organization: int, project: int, repo: RepoDefinition, sha: str | None
    ):
        return CodebaseNamespaceManager.does_repo_exist(
            organization=organization,
            project=project,
            provider=repo.provider,
            external_id=repo.external_id,
            sha=sha,
        )

    @classmethod
    def from_repo_definition(
        cls,
        organization: int,
        project: int,
        repo: RepoDefinition,
        sha: str | None,
        tracking_branch: str | None,
        state: State,
        state_manager_class: Type[CodebaseStateManager],
        embedding_model: SentenceTransformer,
    ):
        logger.debug(f"Loading workspace for {repo.full_name} ({sha or tracking_branch})")
        workspace = CodebaseNamespaceManager.load_workspace_for_repo_definition(
            organization=organization,
            project=project,
            repo=repo,
            sha=sha,
            tracking_branch=tracking_branch,
        )

        logger.debug(
            f"Loaded workspace for {repo.full_name} ({sha or tracking_branch})"
            if workspace
            else f"Failed to load workspace for {organization}/{project}/{repo.external_id} (repo: {repo.full_name} {sha or tracking_branch})"
        )

        if workspace:
            repo_client = RepoClient.from_repo_definition(repo, "read")
            return cls(
                organization,
                project,
                repo_client,
                workspace=workspace,
                state_manager=state_manager_class(workspace.repo_info.id, state),
                embedding_model=embedding_model,
            )

        return None

    @classmethod
    def from_repo_id(
        cls,
        repo_id: int,
        embedding_model: SentenceTransformer,
        state: State | None = None,
        state_manager_class: Type[CodebaseStateManager] | None = None,
        namespace_id: int | None = None,
    ):
        repo_info = cls.get_repo_info_from_db(repo_id)

        if repo_info:
            namespace_id = namespace_id or repo_info.default_namespace

            if not namespace_id:
                raise ValueError(f"Repository with id {repo_id} does not have a default namespace")

            workspace = CodebaseNamespaceManager.load_workspace(namespace_id=namespace_id)

            if workspace:
                state_manager = (
                    state_manager_class(workspace.repo_info.id, state)
                    if state_manager_class and state
                    else DummyCodebaseStateManager()
                )
                return cls(
                    organization=repo_info.organization,
                    project=repo_info.project,
                    repo_client=RepoClient.from_repo_info(repo_info, "read"),
                    workspace=workspace,
                    state_manager=state_manager,
                    embedding_model=embedding_model,
                )

        return None

    @staticmethod
    def create(
        organization: int,
        project: int,
        repo: RepoDefinition,
        tracking_branch: str | None = None,
        sha: str | None = None,
    ) -> int:
        repo_client = RepoClient.from_repo_definition(repo, "read")

        head_sha = sha
        branch = None
        is_default_branch = False

        if tracking_branch:
            branch = tracking_branch
            head_sha = repo_client.get_branch_head_sha(branch)
        elif not head_sha:
            is_default_branch = True
            branch = repo_client.get_default_branch()
            head_sha = repo_client.get_branch_head_sha(branch)

        if not head_sha:
            raise ValueError("Failed to get head sha")

        workspace = CodebaseNamespaceManager.create_namespace_with_new_or_existing_repo(
            organization=organization,
            project=project,
            repo=repo,
            head_sha=head_sha,
            tracking_branch=branch,
            should_set_as_default=is_default_branch,
        )

        return workspace.namespace.id

    @classmethod
    @ai_track(description="Autofix - Indexing namespace")
    def index(
        cls,
        namespace_id: int,
        embedding_model: SentenceTransformer,
        state: State | None = None,
        state_manager_class: Type[CodebaseStateManager] | None = None,
    ):
        workspace = CodebaseNamespaceManager.load_workspace(namespace_id, skip_copy=True)

        if not workspace:
            raise InitializationError("Failed to load workspace for namespace_id")

        repo_client = RepoClient.from_repo_info(workspace.repo_info, "read")

        # If the workspace is ready then we shouldn't index again...
        if not workspace.is_ready():
            try:
                tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir(workspace.namespace.sha)
                logger.debug(f"Loaded repository to {tmp_repo_dir}")

                try:
                    files = repo_client.get_index_file_set(
                        workspace.namespace.sha, skip_empty_files=True
                    )
                    documents = read_specific_files(tmp_repo_dir, files)

                    logger.debug(f"Read {len(documents)} documents:")
                    documents_by_language = group_documents_by_language(documents)
                    for language, docs in documents_by_language.items():
                        logger.debug(f"  {language}: {len(docs)}")

                    doc_parser = DocumentParser(embedding_model)
                    with sentry_sdk.start_span(
                        op="seer.automation.codebase.create.process_documents"
                    ):
                        chunks = doc_parser.process_documents(documents)
                    with sentry_sdk.start_span(op="seer.automation.codebase.create.embed_chunks"):
                        embedded_chunks = cls.embed_chunks(chunks, embedding_model)
                    logger.debug(f"Processed {len(chunks)} chunks")

                    workspace.insert_chunks(embedded_chunks)
                    workspace.namespace.status = CodebaseNamespaceStatus.CREATED
                    workspace.save()

                    logger.debug(f"Create Step: Inserted {len(chunks)} chunks into the database")
                finally:
                    cleanup_dir(tmp_dir)
            except Exception as e:
                logger.error(f"Failed to create codebase index: {e}")

                try:
                    workspace.delete()
                except Exception as ex:
                    sentry_sdk.capture_exception(ex)

                raise e

        return cls(
            workspace.repo_info.organization,
            workspace.repo_info.project,
            repo_client,
            workspace=workspace,
            state_manager=(
                state_manager_class(repo_id=workspace.repo_info.id, state=state)
                if state and state_manager_class
                else DummyCodebaseStateManager()
            ),
            embedding_model=embedding_model,
        )

    def save(self):
        self.workspace.save()

    @ai_track(description="Autofix - Updating codebase index")
    def update(self, sha: str | None = None):
        """
        Updates the codebase index to the latest state of the default branch if needed
        """
        if not self.repo_info:
            raise ValueError("Repository info is not set")

        target_sha = None
        if self.workspace.namespace.tracking_branch:
            target_sha = self.repo_client.get_branch_head_sha(
                self.workspace.namespace.tracking_branch
            )

        elif sha:
            target_sha = sha

        if not target_sha:
            raise ValueError("Provide a sha or run update on a namespace tracking a branch")

        if self.workspace.namespace.sha == target_sha:
            logger.info("Codebase index is up to date")
            return

        changed_files, removed_files = self.repo_client.get_commit_file_diffs(
            self.workspace.namespace.sha, target_sha
        )

        if not changed_files and not removed_files:
            logger.info("No changes to update")
            return
        logger.info(
            f"Updating codebase index with {len(changed_files)} changed files and {len(removed_files)} removed files..."
        )

        tmp_dir, tmp_repo_dir = self.repo_client.load_repo_to_tmp_dir(target_sha)
        logger.debug(f"Loaded repository to {tmp_repo_dir}")

        self.workspace.namespace.status = CodebaseNamespaceStatus.UPDATING
        self.workspace.save()

        try:
            documents = read_specific_files(tmp_repo_dir, changed_files)

            doc_parser = DocumentParser(self.embedding_model)

            with sentry_sdk.start_span(op="seer.automation.codebase.update.process_documents"):
                chunks = doc_parser.process_documents(documents)

            existing_chunks = self.workspace.get_chunks_for_paths(changed_files)

            db_chunk_hashes = set(chunk.hash for chunk in existing_chunks)
            new_chunk_hashes = set([chunk.hash for chunk in chunks])

            chunks_hashes_that_no_longer_exist: set[str] = set()
            for chunk_result in existing_chunks:
                if chunk_result.hash not in new_chunk_hashes:
                    chunks_hashes_that_no_longer_exist.add(chunk_result.hash)

            chunks_to_add: list[BaseDocumentChunk] = []
            chunks_to_update: list[BaseDocumentChunk] = []
            for chunk in chunks:
                if chunk.hash not in db_chunk_hashes:
                    chunks_to_add.append(chunk)
                else:
                    chunks_to_update.append(chunk)

            with sentry_sdk.start_span(op="seer.automation.codebase.update.embed_chunks"):
                embedded_chunks_to_add = self.embed_chunks(chunks_to_add, self.embedding_model)
            logger.debug(f"Processed {len(chunks)} chunks")

            self.workspace.delete_chunks(list(chunks_hashes_that_no_longer_exist))
            self.workspace.update_chunks_metadata(chunks_to_update)
            self.workspace.insert_chunks(embedded_chunks_to_add)
            self.workspace.delete_paths(removed_files)

            self.workspace.namespace.sha = target_sha
            self.workspace.save()

            if not self.verify_file_integrity():
                # Let's see how often this happens, if at all.
                sentry_sdk.capture_message(
                    f"File integrity check after update failed for {self.repo_info.external_slug}, namespace {self.namespace.id}"
                )

            logger.debug(f"Update step: Inserted {len(chunks)} chunks into the database")
        finally:
            self.workspace.namespace.status = CodebaseNamespaceStatus.CREATED
            self.workspace.save_records()

            cleanup_dir(tmp_dir)

    @classmethod
    def embed_chunks(
        cls, chunks: list[BaseDocumentChunk], embedding_model: SentenceTransformer
    ) -> list[EmbeddedDocumentChunk]:
        logger.debug(f"Embedding {len(chunks)} chunks...")
        embeddings_list: list[np.ndarray] = []

        for i in range(0, len(chunks), superchunk_size := 256):
            batch_embeddings: np.ndarray = embedding_model.encode(
                [chunk.get_dump_for_embedding() for chunk in chunks[i : i + superchunk_size]],
                batch_size=4,  # Batch size of 24 works best on a2-ultragpu-1g instance.
                show_progress_bar=False,
            )
            embeddings_list.extend(batch_embeddings)

        embeddings = np.array(embeddings_list)
        logger.debug(f"Embedded {len(chunks)} chunks")

        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunks.append(
                EmbeddedDocumentChunk(
                    **chunk.model_dump(),
                    embedding=embeddings[i],
                )
            )

        torch.cuda.empty_cache()  # TODO: revisit - explicitly including this not a best practice

        return embedded_chunks

    def is_behind(self):
        if not self.repo_info:
            raise ValueError("Repository info is not set")

        if self.namespace.tracking_branch:
            head_sha = self.repo_client.get_branch_head_sha(self.namespace.tracking_branch)
            return self.repo_client.compare(self.namespace.sha, head_sha).ahead_by > 0

        return False

    def verify_file_integrity(self) -> bool:
        """
        Checks if the files in the workspace match the files in the repository
        Note: Only checks up to 100k files for now.
        """
        file_paths = self.repo_client.get_index_file_set(self.namespace.sha, skip_empty_files=True)

        with sentry_sdk.start_span(op="seer.automation.codebase.verify_file_integrity"):
            return self.workspace.verify_file_integrity(file_paths)

    def query(self, query: str, top_k: int = 4) -> list[QueryResultDocumentChunk]:
        assert self.repo_info is not None, "Repository info is not set"

        embedding = self.embedding_model.encode(query, show_progress_bar=False)

        query_results = self.workspace.query_chunks(embedding, top_k)

        return self._get_chunks(query_results)

    @class_method_lru_cache(maxsize=32)
    def _get_file_content_with_cache(self, path: str, sha: str):
        try:
            return self.repo_client.get_file_content(path, sha)
        except Exception:
            return None

    def _copy_document_with_local_changes(
        self, document: DraftDocument | Document
    ) -> Document | None:
        content: str | None = document.text
        # Make sure the changes are applied in order!
        changes = list(
            filter(lambda x: x.path == document.path, self.state_manager.get_file_changes())
        )
        if changes:
            for change in changes:
                content = change.apply(content)

        if content is None or content == "":
            return None

        return Document(path=document.path, text=content, language=document.language)

    def get_document(self, path: str, ignore_local_changes=False) -> Document | None:
        assert self.repo_info is not None, "Repository info is not set"

        document_content = self._get_file_content_with_cache(path, self.namespace.sha)

        language = get_language_from_path(path)
        if language is None:
            logger.warning(f"Unsupported language for {path}")
            return None

        if document_content is None:
            if ignore_local_changes:
                return None
            return self._copy_document_with_local_changes(
                DraftDocument(path=path, language=language)
            )

        document = Document(path=path, text=document_content, language=language)

        if ignore_local_changes:
            return document
        return self._copy_document_with_local_changes(document)

    def store_file_change(self, file_change: FileChange):
        self.state_manager.store_file_change(file_change)

        document = None
        if file_change.change_type != "create":
            document = self.get_document(file_change.path)
            if document is None:
                logger.warning(
                    f"Failed to get document for {file_change.path} when storing file change..."
                )
                return
        else:
            document = Document(
                path=file_change.path,
                text="",
                language=get_language_from_path(file_change.path) or "unknown",
            )

        new_content = file_change.apply(document.text)

        self.workspace.delete_paths([document.path])

        if new_content is not None:
            document.text = new_content

            doc_parser = DocumentParser(self.embedding_model)
            doc_chunks = doc_parser.process_document(document)

            embedded_chunks = self.embed_chunks(doc_chunks, self.embedding_model)

            self.workspace.insert_chunks(embedded_chunks)

    def cleanup(self):
        self.workspace.cleanup()

    def process_stacktrace(self, stacktrace: Stacktrace):
        valid_file_paths = self.repo_client.get_valid_file_paths(self.namespace.sha)
        for frame in stacktrace.frames:
            if frame.in_app and frame.repo_id is None:
                if frame.filename in valid_file_paths:
                    frame.repo_id = self.repo_info.id
                    frame.repo_name = self.repo_info.external_slug
                else:
                    for valid_path in valid_file_paths:
                        if potential_frame_match(valid_path, frame):
                            frame.repo_id = self.repo_info.id
                            frame.repo_name = self.repo_info.external_slug
                            frame.filename = valid_path
                            break

    def _get_chunks(self, chunk_results: list[ChunkQueryResult]) -> list[QueryResultDocumentChunk]:
        # This seems awfully wasteful to chunk and hash a document for each returned chunk but I guess we are offloading the work to when it's needed?
        assert self.repo_info is not None, "Repository info is not set"

        doc_parser = DocumentParser(self.embedding_model)

        matched_chunks: list[QueryResultDocumentChunk] = []
        for chunk_result in chunk_results:
            document = self.get_document(chunk_result.path)

            if document is None:
                logger.warning(f"Failed to get content for {chunk_result.path}")
                # TODO: How to handle this?
                continue

            doc_chunks = doc_parser.process_document(document)
            matched_chunk = next((c for c in doc_chunks if c.hash == chunk_result.hash), None)

            if matched_chunk is None:
                logger.warning(f"Failed to match chunk with hash {chunk_result.hash}")
                continue

            matched_chunk.repo_name = self.repo_info.external_slug
            matched_chunks.append(
                QueryResultDocumentChunk.model_validate(
                    dict(**dict(matched_chunk), distance=chunk_result.distance)
                )
            )

        return matched_chunks

    def get_file_patches(self) -> tuple[list[FilePatch], str]:
        document_paths = list(
            set([file_change.path for file_change in self.state_manager.get_file_changes()])
        )

        if not document_paths:
            return [], ""

        original_documents = [
            self.get_document(path, ignore_local_changes=True) for path in document_paths
        ]
        changed_documents_map: dict[str, Document] = {}

        diffs: list[str] = []
        for i, document in enumerate(original_documents):
            if document and document.text:
                changed_document = self._copy_document_with_local_changes(document)

                diff = difflib.unified_diff(
                    document.text.splitlines(),
                    changed_document.text.splitlines() if changed_document else "",
                    fromfile=document.path,
                    tofile=changed_document.path if changed_document else "/dev/null",
                    lineterm="",
                )

                diff_str = "\n".join(diff).strip("\n")
                diffs.append(diff_str)

                if changed_document:
                    changed_documents_map[changed_document.path] = changed_document
            else:
                path = document_paths[i]
                changed_document = self._copy_document_with_local_changes(
                    Document(path=path, language=get_language_from_path(path) or "unknown", text="")
                )

                if changed_document:
                    diff = difflib.unified_diff(
                        [],  # Empty list to represent no original content
                        changed_document.text.splitlines(),
                        fromfile="/dev/null",
                        tofile=path,
                        lineterm="",
                    )

                    diff_str = "\n".join(diff).strip("\n")
                    diffs.append(diff_str)
                    changed_documents_map[path] = changed_document

        file_patches = []
        for file_diff in diffs:
            patches = PatchSet(file_diff)
            if not patches:
                sentry_sdk.capture_message(f"No patches for diff: {file_diff}")
                continue
            patched_file = patches[0]

            tree: Tree | None = None
            document = changed_documents_map.get(patched_file.path)
            if document and supports_parent_declarations(document.language):
                ast_parser = tree_sitter_languages.get_parser(document.language)
                tree = ast_parser.parse(document.text.encode("utf-8"))

            hunks: list[Hunk] = []
            for hunk in patched_file:
                lines: list[Line] = []
                for line in hunk:
                    lines.append(
                        Line(
                            source_line_no=line.source_line_no,
                            target_line_no=line.target_line_no,
                            diff_line_no=line.diff_line_no,
                            value=line.value,
                            line_type=line.line_type,
                        )
                    )

                section_header = hunk.section_header
                if tree and document:
                    line_numbers = [
                        line.target_line_no
                        for line in lines
                        if line.line_type != " " and line.target_line_no is not None
                    ]
                    first_line_no = line_numbers[0] if line_numbers else None
                    last_line_no = line_numbers[-1] if line_numbers else None
                    if first_line_no is not None and last_line_no is not None:
                        node = tree.root_node.descendant_for_point_range(
                            (first_line_no, 0), (last_line_no, 0)
                        )
                        if node:
                            parent_declaration_node = find_first_parent_declaration(
                                node, document.language
                            )
                            declaration = (
                                extract_declaration(
                                    parent_declaration_node, tree.root_node, document.language
                                )
                                if parent_declaration_node
                                else None
                            )
                            section_header_str = (
                                declaration.to_str(tree.root_node, include_indent=False)
                                if declaration
                                else ""
                            )
                            if section_header_str:
                                section_header_lines = section_header_str.splitlines()
                                if section_header_lines:
                                    section_header = section_header_lines[0].strip()

                hunks.append(
                    Hunk(
                        source_start=hunk.source_start,
                        source_length=hunk.source_length,
                        target_start=hunk.target_start,
                        target_length=hunk.target_length,
                        section_header=section_header,
                        lines=lines,
                    )
                )
            patch_type = (
                patched_file.is_added_file and "A" or patched_file.is_removed_file and "D" or "M"
            )
            file_patches.append(
                FilePatch(
                    type=patch_type,
                    path=patched_file.path,
                    added=patched_file.added,
                    removed=patched_file.removed,
                    source_file=patched_file.source_file,
                    target_file=patched_file.target_file,
                    hunks=hunks,
                )
            )

        combined_diff = "\n".join(diffs)

        return file_patches, combined_diff

    def diff_contains_stacktrace_files(self, event_details: EventDetails) -> bool:
        stacktraces = [exception.stacktrace for exception in event_details.exceptions] + [
            thread.stacktrace for thread in event_details.threads if thread.stacktrace
        ]

        stacktrace_files: set[str] = set()
        for stacktrace in stacktraces:
            for frame in stacktrace.frames:
                if frame.filename:
                    stacktrace_files.add(frame.filename)

        changed_files, removed_files = self.repo_client.get_commit_file_diffs(
            self.namespace.sha, self.repo_client.get_default_branch_head_sha()
        )

        change_files = set(changed_files + removed_files)

        return bool(change_files.intersection(stacktrace_files))
