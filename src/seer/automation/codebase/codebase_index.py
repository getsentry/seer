import logging
import os
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer

from seer.automation.autofix.models import FileChange
from seer.automation.autofix.utils import get_torch_device
from seer.automation.codebase.models import Document, DocumentChunkWithEmbedding
from seer.automation.codebase.parser import DocumentParser
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.utils import cleanup_dir, read_directory, read_specific_files
from seer.db import DbDocumentChunk, RepoInfo, Session
from seer.utils import class_method_lru_cache

logger = logging.getLogger("autofix")


class CodebaseIndex:
    # embedding_model = SentenceTransformer(
    #     os.path.join("./", "models", "issue_grouping_v0/embeddings"), trust_remote_code=True
    # ).to(get_torch_device())

    def __init__(
        self, organization: int, project: int, repo_client: RepoClient, repo_info: RepoInfo | None
    ):
        self.repo_client = repo_client
        self.organization = organization
        self.project = project
        self.repo_info = repo_info
        self.run_id = uuid.uuid4()
        self.file_changes: list[FileChange] = []
        self.embedding_model = SentenceTransformer(
            os.path.join("./", "models", "issue_grouping_v0/embeddings")
        ).to(get_torch_device())

        logger.info(
            f"Loaded codebase index for {repo_client.repo.full_name}, {'with existing data' if self.repo_info else 'without existing data'}"
        )

    @classmethod
    def from_repo_client(cls, organization: int, project: int, repo_client: RepoClient):
        repo_info = (
            Session()
            .query(RepoInfo)
            .filter(
                RepoInfo.organization == organization,
                RepoInfo.project == project,
                RepoInfo.provider == repo_client.provider,
                RepoInfo.external_slug == repo_client.repo.full_name,
            )
            .one_or_none()
        )
        return cls(
            organization,
            project,
            repo_client,
            repo_info,
        )

    @classmethod
    def from_repo_id(cls, repo_id: int):
        repo_info = Session().get(RepoInfo, repo_id)

        if repo_info:
            repo_owner, repo_name = repo_info.external_slug.split("/")

            return cls(
                repo_info.organization,
                repo_info.project,
                RepoClient(repo_info.provider, repo_owner, repo_name),
                repo_info,
            )

        raise ValueError(f"Repository with id {repo_id} not found")

    def create(self):
        head_sha = self.repo_client.get_default_branch_head_sha()
        tmp_dir, tmp_repo_dir = self.repo_client.load_repo_to_tmp_dir(head_sha)
        logger.debug(f"Loaded repository to {tmp_repo_dir}")
        try:
            documents = read_directory(tmp_repo_dir, [".py"])

            logger.debug(f"Read {len(documents)} documents")

            doc_parser = DocumentParser(self.embedding_model)
            chunks = doc_parser.process_documents(documents)
            logger.debug(f"Processed {len(chunks)} chunks")

            with Session() as session:
                repo_info = RepoInfo(
                    organization=self.organization,
                    project=self.project,
                    provider=self.repo_client.provider,
                    external_slug=self.repo_client.repo.full_name,
                    sha=head_sha,
                )
                session.add(repo_info)
                session.flush()
                logger.debug(f"Inserted repository info with id {repo_info.id}")
                self.repo_info = repo_info

                db_chunks = [chunk.to_db_model(repo_info.id) for chunk in chunks]
                session.add_all(db_chunks)
                session.commit()

            logger.debug(f"Create Step: Inserted {len(chunks)} chunks into the database")
        finally:
            cleanup_dir(tmp_dir)

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

            doc_parser = DocumentParser(self.embedding_model)
            chunks = doc_parser.process_documents(documents)
            logger.debug(f"Processed {len(chunks)} chunks")

            with Session() as session:
                db_chunks = [chunk.to_db_model(self.repo_info.id) for chunk in chunks]
                session.add_all(db_chunks)

                if removed_files:
                    session.query(DbDocumentChunk).filter(
                        DbDocumentChunk.repository_id == self.repo_info.id,
                        DbDocumentChunk.path.in_(removed_files),
                    ).delete(synchronize_session=False)

                self.repo_info.sha = head_sha
                repo_info = session.get(RepoInfo, self.repo_info.id)
                if repo_info is None:
                    raise ValueError(f"Repository info with id {self.repo_info.id} not found")
                repo_info.sha = head_sha

                session.commit()

            logger.debug(f"Update step: Inserted {len(chunks)} chunks into the database")
        finally:
            cleanup_dir(tmp_dir)

    def is_behind(self):
        if not self.repo_info:
            raise ValueError("Repository info is not set")

        head_sha = self.repo_client.get_default_branch_head_sha()

        return self.repo_client.compare(self.repo_info.sha, head_sha).ahead_by > 0

    def query(self, query: str, top_k: int = 4):
        assert self.repo_info is not None, "Repository info is not set"

        embedding = self.embedding_model.encode(query, show_progress_bar=False)

        with Session() as session:
            db_chunks = (
                session.query(DbDocumentChunk)
                .filter(
                    DbDocumentChunk.repository_id == self.repo_info.id,
                    (DbDocumentChunk.for_run_id == self.run_id)
                    | (DbDocumentChunk.for_run_id.is_(None)),
                )
                .order_by(DbDocumentChunk.embedding.cosine_distance(embedding))
                .limit(top_k)
                .all()
            )

            return self._populate_chunks(db_chunks)

    @class_method_lru_cache(maxsize=32)
    def _get_file_content_with_cache(self, path: str, sha: str):
        return self.repo_client.get_file_content(path, sha)

    def get_document(self, path: str, ignore_local_changes=False) -> Document | None:
        assert self.repo_info is not None, "Repository info is not set"

        document_content = self._get_file_content_with_cache(path, self.repo_info.sha)

        if document_content is None:
            return None

        document = Document(path=path, text=document_content)

        content = document_content
        if not ignore_local_changes:
            # Make sure the changes are applied in order!
            changes = list(filter(lambda x: x.path == path, self.file_changes))
            if changes:
                for change in changes:
                    content = change.apply(content)

            if content is None:
                return None

        document.text = content

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
                DbDocumentChunk.repository_id == self.repo_info.id,
                DbDocumentChunk.path == document.path,
                DbDocumentChunk.for_run_id == str(self.run_id),
            ).delete(synchronize_session=False)

            doc_parser = DocumentParser(self.embedding_model)
            chunks = doc_parser.process_document(document)

            db_chunks = []
            for chunk in chunks:
                db_chunk = chunk.to_db_model(self.repo_info.id)
                db_chunk.for_run_id = str(self.run_id)
            session.add_all(db_chunks)
            session.commit()

    def cleanup(self):
        with Session() as session:
            rows_to_delete = session.query(DbDocumentChunk).filter(
                DbDocumentChunk.for_run_id == str(self.run_id)
            )
            rows_to_delete.delete(synchronize_session=False)
            session.commit()

            logger.debug(
                f"Deleted {rows_to_delete.count()} rows from document_chunks for run_id {self.run_id}"
            )

        logger.info(f"Cleaned up temporary data for run_id {self.run_id}")

    def _populate_chunks(self, chunks: list[DbDocumentChunk]) -> list[DocumentChunkWithEmbedding]:
        ### This seems awfully wasteful to chunk and hash a document for each returned chunk but I guess we are offloading the work to when it's needed?
        assert self.repo_info is not None, "Repository info is not set"

        doc_parser = DocumentParser(self.embedding_model)

        matched_chunks: list[DocumentChunkWithEmbedding] = []
        for chunk in chunks:
            content = self._get_file_content_with_cache(chunk.path, self.repo_info.sha)

            if content is None:
                logger.warning(f"Failed to get content for {chunk.path}")
                # TODO: How to handle this?
                continue

            doc_chunks = doc_parser.process_document(Document(path=chunk.path, text=content))
            matched_chunk = next((c for c in doc_chunks if c.hash == chunk.hash), None)

            if matched_chunk is None:
                logger.warning(f"Failed to match chunk with hash {chunk.hash}")
                continue

            matched_chunks.append(
                DocumentChunkWithEmbedding(
                    id=chunk.id,
                    path=chunk.path,
                    index=chunk.index,
                    hash=chunk.hash,
                    token_count=chunk.token_count,
                    first_line_number=chunk.first_line_number,
                    embedding=np.array(chunk.embedding),
                    content=matched_chunk.content,
                    context=matched_chunk.context,
                )
            )

        return matched_chunks
