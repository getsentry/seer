import logging
import uuid

import numpy as np
from langsmith import RunTree, traceable
from tqdm import tqdm

from seer.automation.autofix.models import FileChange, RepoDefinition, Stacktrace
from seer.automation.codebase.models import (
    Document,
    DocumentChunk,
    DocumentChunkWithEmbedding,
    DocumentChunkWithEmbeddingAndId,
    RepositoryInfo,
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
from seer.automation.utils import get_embedding_model
from seer.db import DbDocumentChunk, DbRepositoryInfo, Session
from seer.utils import class_method_lru_cache

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
                logger.debug(f"Inserted repository info with id {db_repo_info.id}")

                documents = read_directory(tmp_repo_dir, repo_id=db_repo_info.id)

                logger.debug(f"Read {len(documents)} documents:")
                documents_by_language = group_documents_by_language(documents)
                for language, docs in documents_by_language.items():
                    logger.debug(f"  {language}: {len(docs)}")

                doc_parser = DocumentParser(get_embedding_model())
                chunks = doc_parser.process_documents(documents)
                embedded_chunks = cls._embed_chunks(chunks)
                logger.debug(f"Processed {len(chunks)} chunks")

                repo_info = RepositoryInfo.from_db(db_repo_info)

                db_chunks = [chunk.to_db_model() for chunk in embedded_chunks]
                session.add_all(db_chunks)
                session.commit()

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
            documents = read_specific_files(tmp_repo_dir, changed_files, repo_id=self.repo_info.id)

            doc_parser = DocumentParser(get_embedding_model())
            chunks = doc_parser.process_documents(documents)
            embedded_chunks = self._embed_chunks(chunks)
            logger.debug(f"Processed {len(chunks)} chunks")

            with Session() as session:
                db_chunks = [chunk.to_db_model() for chunk in embedded_chunks]
                session.add_all(db_chunks)

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
    def _embed_chunks(cls, chunks: list[DocumentChunk]) -> list[DocumentChunkWithEmbedding]:
        logger.debug(f"Embedding {len(chunks)} chunks...")
        embeddings_list: list[np.ndarray] = []

        with tqdm(total=len(chunks)) as pbar:
            for i in range(0, len(chunks), superchunk_size := 128):
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
                DocumentChunkWithEmbedding(
                    **chunk.model_dump(),
                    embedding=embeddings[i],
                )
            )

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

    def get_document(self, path: str, ignore_local_changes=False) -> Document | None:
        assert self.repo_info is not None, "Repository info is not set"

        document_content = self._get_file_content_with_cache(path, self.repo_info.sha)

        if document_content is None:
            return None

        language = get_language_from_path(path)

        if language is None:
            logger.warning(f"Unsupported language for {path}")
            return None

        document = Document(
            path=path, text=document_content, repo_id=self.repo_info.id, language=language
        )

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
                DbDocumentChunk.repo_id == self.repo_info.id,
                DbDocumentChunk.path == document.path,
                DbDocumentChunk.namespace == str(self.run_id),
            ).delete(synchronize_session=False)

            doc_parser = DocumentParser(get_embedding_model())
            chunks = doc_parser.process_document(document)
            embedded_chunks = self._embed_chunks(chunks)

            db_chunks: list[DbDocumentChunk] = []
            for chunk in embedded_chunks:
                db_chunk = chunk.to_db_model()
                db_chunk.namespace = str(self.run_id)
            session.add_all(db_chunks)
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

    def _populate_chunks(
        self, chunks: list[DbDocumentChunk]
    ) -> list[DocumentChunkWithEmbeddingAndId]:
        ### This seems awfully wasteful to chunk and hash a document for each returned chunk but I guess we are offloading the work to when it's needed?
        assert self.repo_info is not None, "Repository info is not set"

        doc_parser = DocumentParser(get_embedding_model())

        matched_chunks: list[DocumentChunkWithEmbeddingAndId] = []
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
                    repo_id=self.repo_info.id,
                    language=chunk.language,
                )
            )
            matched_chunk = next((c for c in doc_chunks if c.hash == chunk.hash), None)

            if matched_chunk is None:
                logger.warning(f"Failed to match chunk with hash {chunk.hash}")
                continue

            matched_chunks.append(
                DocumentChunkWithEmbeddingAndId(
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
