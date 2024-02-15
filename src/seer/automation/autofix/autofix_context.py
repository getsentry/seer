import uuid

from sentence_transformers.util import cos_sim

from seer.automation.autofix.models import RepoDefinition, Stacktrace
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import DocumentChunkWithEmbedding
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.utils import get_embedding_model
from seer.db import DbDocumentChunk, Session


class AutofixContext:
    codebases: dict[int, CodebaseIndex]

    def __init__(
        self,
        organization_id: int,
        project_id: int,
        repos: list[RepoDefinition],
    ):
        self.organization_id = organization_id
        self.project_id = project_id
        self.run_id = uuid.uuid4()
        self.codebases = {}

        for repo in repos:
            codebase_index = CodebaseIndex.from_repo_definition(
                organization_id, project_id, repo, self.run_id
            )

            if codebase_index:
                self.codebases[codebase_index.repo_info.id] = codebase_index

    def has_codebase_index(self, repo: RepoDefinition) -> bool:
        return CodebaseIndex.has_repo_been_indexed(self.organization_id, self.project_id, repo)

    def create_codebase_index(self, repo: RepoDefinition):
        codebase_index = CodebaseIndex.create(
            self.organization_id, self.project_id, repo, self.run_id
        )
        self.codebases[codebase_index.repo_info.id] = codebase_index

    def get_codebase(self, repo_id: int) -> CodebaseIndex:
        codebase = self.codebases[repo_id]

        if codebase is None:
            raise ValueError(f"Codebase with id {repo_id} not found")

        return codebase

    def query(self, query: str, top_k: int = 8):
        repo_ids = list(self.codebases.keys())

        embedding = get_embedding_model().encode(query)

        with Session() as session:
            db_chunks = (
                session.query(DbDocumentChunk)
                .filter(
                    DbDocumentChunk.repo_id.in_(repo_ids),
                    (DbDocumentChunk.for_run_id == str(self.run_id))
                    | (DbDocumentChunk.for_run_id.is_(None)),
                )
                .order_by(DbDocumentChunk.embedding.cosine_distance(embedding))
                .limit(top_k)
                .all()
            )

            chunks_by_repo_id = {}
            for db_chunk in db_chunks:
                chunks_by_repo_id.setdefault(db_chunk.repo_id, []).append(db_chunk)

            populated_chunks = []
            for repo_id, db_chunks in chunks_by_repo_id.items():
                codebase = self.get_codebase(repo_id)
                populated_chunks.extend(codebase._populate_chunks(db_chunks))

            # Re-sort populated_chunks based on their original order in db_chunks
            db_chunk_order = {db_chunk.id: index for index, db_chunk in enumerate(db_chunks)}
            populated_chunks.sort(key=lambda chunk: db_chunk_order[chunk.id])

        return populated_chunks

    def diff_contains_stacktrace_files(self, repo_id: int, stacktrace: Stacktrace) -> bool:
        codebase = self.get_codebase(repo_id)
        changed_files, removed_files = codebase.repo_client.get_commit_file_diffs(
            codebase.repo_info.sha, codebase.repo_client.get_default_branch_head_sha()
        )

        change_files = set(changed_files + removed_files)
        stacktrace_files = set([frame.filename for frame in stacktrace.frames])

        return bool(change_files.intersection(stacktrace_files))

    def annotate_stacktrace_with_repo(self, stacktrace: Stacktrace):
        for frame in stacktrace.frames:
            for codebase in self.codebases.values():
                document = codebase.get_document(frame.filename)
                if document:
                    frame.repo_name = codebase.repo_info.external_slug

    def cleanup(self):
        for codebase in self.codebases.values():
            codebase.cleanup()
