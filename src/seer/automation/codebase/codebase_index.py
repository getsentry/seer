import difflib
import logging
import uuid

import numpy as np
import sentry_sdk
import torch
import tree_sitter_languages
from langsmith import traceable
from sqlalchemy.orm import class_mapper
from tqdm import tqdm
from tree_sitter import Tree
from unidiff import PatchSet

from seer.automation.autofix.models import RepoDefinition, Stacktrace
from seer.automation.codebase.ast import (
    extract_declaration,
    find_first_parent_declaration,
    supports_parent_declarations,
)
from seer.automation.codebase.models import (
    BaseDocumentChunk,
    Document,
    EmbeddedDocumentChunk,
    RepositoryInfo,
    StoredDocumentChunk,
)
from seer.automation.codebase.parser import DocumentParser
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.utils import (
    cleanup_dir,
    get_language_from_path,
    group_documents_by_language,
    potential_frame_match,
    read_directory,
    read_specific_files,
)
from seer.automation.models import FileChange, FilePatch, Hunk, Line
from seer.automation.utils import get_embedding_model
from seer.db import DbDocumentChunk, DbRepositoryInfo, Session
from seer.utils import batch_save_to_db, class_method_lru_cache

logger = logging.getLogger("autofix")


class CodebaseIndex:
    def __init__(
        self,
        organization: int,
        project: int,
        repo_client: RepoClient,
        repo_info: RepositoryInfo,
        run_id: uuid.UUID,
    ):
        self.repo_client = repo_client
        self.organization = organization
        self.project = project
        self.repo_info = repo_info
        self.file_changes: list[FileChange] = []
        self.run_id = run_id

        logger.info(
            f"Loaded codebase index for {repo_client.repo.full_name}, {'with existing data' if self.repo_info else 'without existing data'}"
        )

    @staticmethod
    def has_repo_been_indexed(organization: int, project: int, repo: RepoDefinition):
        return (
            Session()
            .query(DbRepositoryInfo)
            .filter(
                DbRepositoryInfo.organization == organization,
                DbRepositoryInfo.project == project,
                DbRepositoryInfo.provider == repo.provider,
                DbRepositoryInfo.external_slug == f"{repo.owner}/{repo.name}",
            )
            .count()
            > 0
        )

    @classmethod
    def from_repo_definition(
        cls, organization: int, project: int, repo: RepoDefinition, run_id: uuid.UUID
    ):
        db_repo_info = (
            Session()
            .query(DbRepositoryInfo)
            .filter(
                DbRepositoryInfo.organization == organization,
                DbRepositoryInfo.project == project,
                DbRepositoryInfo.provider == repo.provider,
                DbRepositoryInfo.external_slug == f"{repo.owner}/{repo.name}",
            )
            .one_or_none()
        )
        if db_repo_info:
            repo_info = RepositoryInfo.from_db(db_repo_info)
            repo_client = RepoClient(repo.provider, repo.owner, repo.name)
            return cls(organization, project, repo_client, repo_info, run_id)

        return None

    @classmethod
    def from_repo_id(cls, repo_id: int):
        db_repo_info = Session().get(DbRepositoryInfo, repo_id)

        if db_repo_info:
            repo_info = RepositoryInfo.from_db(db_repo_info)
            repo_owner, repo_name = db_repo_info.external_slug.split("/")

            return cls(
                repo_info.organization,
                repo_info.project,
                RepoClient(repo_info.provider, repo_owner, repo_name),
                repo_info,
                run_id=uuid.uuid4(),
            )

        raise ValueError(f"Repository with id {repo_id} not found")

    @classmethod
    @traceable(name="Creating codebase index")
    def create(cls, organization: int, project: int, repo: RepoDefinition, run_id: uuid.UUID):
        repo_client = RepoClient(repo.provider, repo.owner, repo.name)

        head_sha = repo_client.get_default_branch_head_sha()
        tmp_dir, tmp_repo_dir = repo_client.load_repo_to_tmp_dir(head_sha)
        logger.debug(f"Loaded repository to {tmp_repo_dir}")
        try:
            documents = read_directory(tmp_repo_dir)

            logger.debug(f"Read {len(documents)} documents:")
            documents_by_language = group_documents_by_language(documents)
            for language, docs in documents_by_language.items():
                logger.debug(f"  {language}: {len(docs)}")

            doc_parser = DocumentParser(get_embedding_model())
            with sentry_sdk.start_span(op="seer.automation.codebase.create.process_documents"):
                chunks = doc_parser.process_documents(documents)
            with sentry_sdk.start_span(op="seer.automation.codebase.create.embed_chunks"):
                embedded_chunks = cls._embed_chunks(chunks)
            logger.debug(f"Processed {len(chunks)} chunks")

            with Session() as session:
                db_repo_info = DbRepositoryInfo(
                    organization=organization,
                    project=project,
                    provider=repo_client.provider,
                    external_slug=repo_client.repo.full_name,
                    sha=head_sha,
                )
                session.add(db_repo_info)
                session.flush()

                db_chunks = [
                    chunk.to_db_model(repo_id=db_repo_info.id) for chunk in embedded_chunks
                ]

                batch_save_to_db(session, db_chunks)

                session.commit()

                repo_info = RepositoryInfo.from_db(db_repo_info)

            logger.debug(f"Create Step: Inserted {len(chunks)} chunks into the database")

            return cls(organization, project, repo_client, repo_info, run_id)
        finally:
            cleanup_dir(tmp_dir)

    @traceable(name="Updating codebase index")
    def update(self):
        """
        Updates the codebase index to the latest state of the default branch if needed
        """
        if not self.repo_info:
            raise ValueError("Repository info is not set")

        head_sha = self.repo_client.get_default_branch_head_sha()
        changed_files, removed_files = self.repo_client.get_commit_file_diffs(
            self.repo_info.sha, head_sha
        )

        if not changed_files and not removed_files:
            logger.info("No changes to update")
            return
        logger.info(
            f"Updating codebase index with {len(changed_files)} changed files and {len(removed_files)} removed files..."
        )

        tmp_dir, tmp_repo_dir = self.repo_client.load_repo_to_tmp_dir(head_sha)
        logger.debug(f"Loaded repository to {tmp_repo_dir}")

        try:
            documents = read_specific_files(tmp_repo_dir, changed_files)

            doc_parser = DocumentParser(get_embedding_model())

            with sentry_sdk.start_span(op="seer.automation.codebase.update.process_documents"):
                chunks = doc_parser.process_documents(documents)

            with Session() as session:
                existing_chunks: list[DbDocumentChunk] = (
                    session.query(DbDocumentChunk)
                    .filter(
                        DbDocumentChunk.repo_id == self.repo_info.id,
                        DbDocumentChunk.path.in_(changed_files),
                    )
                    .all()
                )

            db_chunk_hash_map = {chunk.hash: chunk for chunk in existing_chunks}
            new_chunk_hashes = set([chunk.hash for chunk in chunks])

            chunks_ids_that_no_longer_exist: set[int] = set()
            for db_chunk in existing_chunks:
                if db_chunk.hash not in new_chunk_hashes:
                    chunks_ids_that_no_longer_exist.add(db_chunk.id)

            chunks_to_add: list[BaseDocumentChunk] = []
            chunks_indexes_to_update: list[tuple[int, int]] = []
            for chunk in chunks:
                if chunk.hash not in db_chunk_hash_map:
                    chunks_to_add.append(chunk)
                else:
                    db_chunk = db_chunk_hash_map[chunk.hash]
                    chunks_indexes_to_update.append((db_chunk.id, chunk.index))

            with sentry_sdk.start_span(op="seer.automation.codebase.update.embed_chunks"):
                embedded_chunks_to_add = self._embed_chunks(chunks_to_add)
            logger.debug(f"Processed {len(chunks)} chunks")

            with Session() as session:
                session.query(DbDocumentChunk).filter(
                    DbDocumentChunk.id.in_(chunks_ids_that_no_longer_exist)
                ).delete(synchronize_session=False)

                session.flush()

                # Bulk update indices of the chunks that already exist
                session.bulk_update_mappings(
                    class_mapper(DbDocumentChunk),
                    [
                        {"id": db_chunk_id, "index": new_index}
                        for db_chunk_id, new_index in chunks_indexes_to_update
                    ],
                )

                session.flush()

                new_db_chunks = [
                    chunk.to_db_model(repo_id=self.repo_info.id) for chunk in embedded_chunks_to_add
                ]
                batch_save_to_db(session, new_db_chunks)

                if removed_files:
                    session.query(DbDocumentChunk).filter(
                        DbDocumentChunk.repo_id == self.repo_info.id,
                        DbDocumentChunk.path.in_(removed_files),
                    ).delete(synchronize_session=False)

                self.repo_info.sha = head_sha
                db_repo_info = session.get(DbRepositoryInfo, self.repo_info.id)
                if db_repo_info is None:
                    raise ValueError(f"Repository info with id {self.repo_info.id} not found")
                db_repo_info.sha = head_sha

                session.commit()

            logger.debug(f"Update step: Inserted {len(chunks)} chunks into the database")
        finally:
            cleanup_dir(tmp_dir)

    @classmethod
    def _embed_chunks(cls, chunks: list[BaseDocumentChunk]) -> list[EmbeddedDocumentChunk]:
        logger.debug(f"Embedding {len(chunks)} chunks...")
        embeddings_list: list[np.ndarray] = []

        with tqdm(total=len(chunks)) as pbar:
            for i in range(0, len(chunks), superchunk_size := 64):
                batch_embeddings: np.ndarray = get_embedding_model().encode(
                    [chunk.get_dump_for_embedding() for chunk in chunks[i : i + superchunk_size]],
                    batch_size=4,
                    show_progress_bar=True,
                )
                embeddings_list.extend(batch_embeddings)
                pbar.update(superchunk_size)

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

        head_sha = self.repo_client.get_default_branch_head_sha()

        return self.repo_client.compare(self.repo_info.sha, head_sha).ahead_by > 0

    def query(self, query: str, top_k: int = 4):
        assert self.repo_info is not None, "Repository info is not set"

        embedding = get_embedding_model().encode(query, show_progress_bar=False)

        with Session() as session:
            db_chunks = (
                session.query(DbDocumentChunk)
                .filter(
                    DbDocumentChunk.repo_id == self.repo_info.id,
                    (DbDocumentChunk.namespace == str(self.run_id))
                    | (DbDocumentChunk.namespace.is_(None)),
                )
                .order_by(DbDocumentChunk.embedding.cosine_distance(embedding))
                .limit(top_k)
                .all()
            )

            return self._populate_chunks(db_chunks)

    @class_method_lru_cache(maxsize=32)
    def _get_file_content_with_cache(self, path: str, sha: str):
        return self.repo_client.get_file_content(path, sha)

    def _copy_document_with_local_changes(self, document: Document) -> Document | None:
        content: str | None = document.text
        # Make sure the changes are applied in order!
        changes = list(filter(lambda x: x.path == document.path, self.file_changes))
        if changes:
            for change in changes:
                content = change.apply(content)

        if content is None or content == "":
            return None

        return Document(path=document.path, text=content, language=document.language)

    def get_document(self, path: str, ignore_local_changes=False) -> Document | None:
        assert self.repo_info is not None, "Repository info is not set"

        document_content = self._get_file_content_with_cache(path, self.repo_info.sha)

        if document_content is None:
            return None

        language = get_language_from_path(path)

        if language is None:
            logger.warning(f"Unsupported language for {path}")
            return None

        document = Document(path=path, text=document_content, language=language)

        if not ignore_local_changes:
            return self._copy_document_with_local_changes(document)

        return document

    def store_file_change(self, file_change: FileChange):
        self.file_changes.append(file_change)

        document = self.get_document(file_change.path)
        if document is None:
            logger.warning(
                f"Failed to get document for {file_change.path} when storing file change..."
            )
            return

        new_content = file_change.apply(document.text)

        if new_content is not None:
            self.update_document_temporarily(document)

        # TODO: How to handle deleting documents temporarily?

    def update_document_temporarily(self, document: Document):
        assert self.repo_info is not None, "Repository info is not set"

        with Session() as session:
            # Delete the entire document from the temporary chunks if it exists
            session.query(DbDocumentChunk).filter(
                DbDocumentChunk.repo_id == self.repo_info.id,
                DbDocumentChunk.path == document.path,
                DbDocumentChunk.namespace == str(self.run_id),
            ).delete(synchronize_session=False)

            doc_parser = DocumentParser(get_embedding_model())
            chunks = doc_parser.process_document(document)
            embedded_chunks = self._embed_chunks(chunks)

            db_chunks: list[DbDocumentChunk] = []
            for chunk in embedded_chunks:
                db_chunk = chunk.to_db_model(repo_id=self.repo_info.id)
                db_chunk.namespace = str(self.run_id)

            batch_save_to_db(session, db_chunks)

            session.commit()

    def cleanup(self):
        with Session() as session:
            rows_to_delete = session.query(DbDocumentChunk).filter(
                DbDocumentChunk.namespace == str(self.run_id)
            )
            rows_to_delete.delete(synchronize_session=False)
            session.commit()

            logger.debug(
                f"Deleted {rows_to_delete.count()} rows from document_chunks for run_id {self.run_id}"
            )

        logger.info(f"Cleaned up temporary data for run_id {self.run_id}")

    def process_stacktrace(self, stacktrace: Stacktrace):
        valid_file_paths = self.repo_client.get_valid_file_paths(self.repo_info.sha)
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

    def _populate_chunks(self, chunks: list[DbDocumentChunk]) -> list[StoredDocumentChunk]:
        ### This seems awfully wasteful to chunk and hash a document for each returned chunk but I guess we are offloading the work to when it's needed?
        assert self.repo_info is not None, "Repository info is not set"

        doc_parser = DocumentParser(get_embedding_model())

        matched_chunks: list[StoredDocumentChunk] = []
        for chunk in chunks:
            content = self._get_file_content_with_cache(chunk.path, self.repo_info.sha)

            if content is None:
                logger.warning(f"Failed to get content for {chunk.path}")
                # TODO: How to handle this?
                continue

            doc_chunks = doc_parser.process_document(
                Document(
                    path=chunk.path,
                    text=content,
                    language=chunk.language,
                )
            )
            matched_chunk = next((c for c in doc_chunks if c.hash == chunk.hash), None)

            if matched_chunk is None:
                logger.warning(f"Failed to match chunk with hash {chunk.hash}")
                continue

            matched_chunks.append(
                StoredDocumentChunk(
                    id=chunk.id,
                    path=chunk.path,
                    index=chunk.index,
                    hash=chunk.hash,
                    token_count=chunk.token_count,
                    embedding=np.array(chunk.embedding),
                    content=matched_chunk.content,
                    context=matched_chunk.context,
                    repo_id=chunk.repo_id,
                    language=chunk.language,
                )
            )

        return matched_chunks

    def get_file_patches(self) -> list[FilePatch]:
        document_paths = list(set([file_change.path for file_change in self.file_changes]))

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
                        "",
                        changed_document.text.splitlines(),
                        fromfile="/dev/null",
                        tofile=path,
                        lineterm="",
                    )

                    diff_str = "\n".join(diff).strip("\n")
                    diffs.append(diff_str)
                    changed_documents_map[path] = changed_document

        combined_diff = "\n".join(diffs).strip("\n")
        patch = PatchSet(combined_diff)
        file_patches = []
        for patched_file in patch:
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
                            section_header = (
                                declaration.to_str(tree.root_node, include_indent=False)
                                .splitlines()[0]
                                .strip()
                                if declaration
                                else section_header
                            )

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

        return file_patches
