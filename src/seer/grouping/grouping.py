import gc
import logging
from functools import wraps
from typing import Any, List, Optional

import numpy as np
import sentry_sdk
import sqlalchemy.orm
import torch
from pydantic import BaseModel, ValidationInfo, field_validator
from sentence_transformers import SentenceTransformer
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from torch.cuda import OutOfMemoryError

from seer.db import DbGroupingRecord, Session
from seer.stubs import DummySentenceTransformer, can_use_model_stubs

logger = logging.getLogger(__name__)

NN_GROUPING_DISTANCE = 0.01
NN_GROUPING_HNSW_DISTANCE = 0.05
NN_GROUPING_HNSW_CANDIDATES = 100
NN_SIMILARITY_DISTANCE = 0.1


class GroupingRequest(BaseModel):
    project_id: int
    stacktrace: str
    hash: str
    exception_type: Optional[str] = None
    k: int = 1
    threshold: float = NN_GROUPING_DISTANCE
    read_only: bool = False
    hnsw_candidates: int = NN_GROUPING_HNSW_CANDIDATES
    hnsw_distance: float = NN_GROUPING_HNSW_DISTANCE
    use_reranking: bool = False

    @field_validator("stacktrace")
    @classmethod
    def check_field_is_not_empty(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError(f"{info.field_name} must be provided and not empty.")
        return v


class GroupingResponse(BaseModel):
    parent_hash: str
    stacktrace_distance: float
    should_group: bool


class SimilarityResponse(BaseModel):
    responses: List[GroupingResponse]


class CreateGroupingRecordData(BaseModel):
    group_id: int
    hash: str
    project_id: int
    exception_type: Optional[str] = None


class CreateGroupingRecordsRequest(BaseModel):
    data: List[CreateGroupingRecordData]
    stacktrace_list: List[str]
    encode_stacktrace_batch_size: int = 1
    threshold: float = NN_GROUPING_DISTANCE
    k: int = 1
    hnsw_candidates: int = NN_GROUPING_HNSW_CANDIDATES
    hnsw_distance: float = NN_GROUPING_HNSW_DISTANCE
    use_reranking: bool = False


class BulkCreateGroupingRecordsResponse(BaseModel):
    success: bool
    groups_with_neighbor: dict[str, GroupingResponse]


class DeleteGroupingRecordsByHashRequest(BaseModel):
    project_id: int
    hash_list: List[str]


class DeleteGroupingRecordsByHashResponse(BaseModel):
    success: bool


class GroupingRecord(BaseModel):
    project_id: int
    stacktrace_embedding: np.ndarray
    hash: str
    error_type: Optional[str] = None

    def to_db_model(self) -> DbGroupingRecord:
        return DbGroupingRecord(
            project_id=self.project_id,
            stacktrace_embedding=self.stacktrace_embedding,
            hash=self.hash,
            error_type=self.error_type,
        )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()  # Convert ndarray to list for serialization
        }


def _load_model(model_path: str) -> SentenceTransformer:
    if can_use_model_stubs():
        return DummySentenceTransformer(embedding_size=768)

    model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Loading transformer model to device {model_device}")
    return SentenceTransformer(
        model_path,
        trust_remote_code=True,
        device=model_device,
    )


def handle_out_of_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (OutOfMemoryError, RuntimeError) as e:
            # Only handle CUDA-related RuntimeErrors
            if isinstance(e, RuntimeError) and "CUDA" not in str(e):
                raise
            logger.warning("Ran out of memory, clearing cache and retrying once")
            gc.collect()
            torch.cuda.empty_cache()
            return func(*args, **kwargs)

    return wrapper


class GroupingLookup:
    model: SentenceTransformer

    """
    Manages the grouping of similar stack traces using sentence embeddings and pgvector for similarity search.
    """

    def __init__(self, model_path: str, data_path: str):
        """
        Initializes the GroupingLookup with the sentence transformer model.

        :param model_path: Path to the sentence transformer model.
        """
        self.model = _load_model(model_path)
        self.encode_text("IndexError: list index out of range")  # Ensure warm start

    @sentry_sdk.tracing.trace
    @handle_out_of_memory
    def encode_text(self, stacktrace: str) -> np.ndarray:
        """
        Encodes the stacktrace using the sentence transformer model.

        :param stacktrace: The stacktrace to encode.
        :return: The embedding of the stacktrace.
        """
        return self.model.encode(stacktrace)

    @sentry_sdk.tracing.trace
    @handle_out_of_memory
    def encode_multiple_texts(self, stacktraces: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Encodes multiple stacktraces in batches using the sentence transformer model.
        :param stacktraces: The list of stacktraces to encode.
        :param batch_size: The batch size used for the computation.
        :return: The embeddings of the stacktraces.
        """

        return self.model.encode(sentences=stacktraces, batch_size=batch_size)

    @sentry_sdk.tracing.trace
    def query_nearest_k_neighbors(
        self,
        session: sqlalchemy.orm.Session,
        embedding: np.ndarray,
        project_id: int,
        hash: str,
        distance: float,
        k: int,
        hnsw_candidates: int,
        hnsw_distance: float,
        use_reranking: bool,
    ) -> List[tuple[DbGroupingRecord, float]]:
        """
        Query the nearest k neighbors for a given embedding.

        This method performs a similarity search to find the k nearest neighbors
        for the provided embedding. It can reranking based on the
        `use_reranking` parameter.

        Args:
            session (sqlalchemy.orm.Session): The database session.
            embedding (np.ndarray): The embedding to search for similar records.
            project_id (int): The ID of the project to search within.
            hash (str): The hash to exclude from the search results.
            distance (float): The maximum cosine distance for similarity.
            k (int): The number of nearest neighbors to return.
            hnsw_candidates (int): The number of candidates for HNSW search (used with reranking).
            hnsw_distance (float): The maximum distance for HNSW search (used with reranking).
            use_reranking (bool): Whether to use reranking in the search process.

        Returns:
            List[tuple[DbGroupingRecord, float]]: A list of tuples containing the nearest
            neighbor records and their distances.
        """
        if use_reranking:
            return self._query_with_reranking(
                session, embedding, project_id, hash, distance, k, hnsw_candidates, hnsw_distance
            )
        else:
            return self._query_without_reranking(session, embedding, project_id, hash, distance, k)

    def _query_without_reranking(
        self,
        session: sqlalchemy.orm.Session,
        embedding: np.ndarray,
        project_id: int,
        hash: str,
        distance: float,
        k: int,
    ) -> List[tuple[DbGroupingRecord, float]]:
        custom_options = {"postgresql_execute_before": "SET LOCAL hnsw.ef_search = 100"}

        candidates = (
            session.query(DbGroupingRecord)
            .filter(
                DbGroupingRecord.project_id == project_id,
                DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding) <= distance,
                DbGroupingRecord.hash != hash,
            )
            .order_by(DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding))
            .limit(k)
            .execution_options(**custom_options)
            .all()
        )

        return [
            (candidate, self.cosine_distance(embedding, candidate.stacktrace_embedding))
            for candidate in candidates
        ]

    def _query_with_reranking(
        self,
        session: sqlalchemy.orm.Session,
        embedding: np.ndarray,
        project_id: int,
        hash: str,
        distance: float,
        k: int,
        hnsw_candidates: int,
        hnsw_distance: float,
    ) -> List[tuple[DbGroupingRecord, float]]:
        custom_options = {"postgresql_execute_before": "SET LOCAL hnsw.ef_search = 100"}

        candidates = (
            session.query(DbGroupingRecord)
            .filter(
                DbGroupingRecord.project_id == project_id,
                DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding)
                <= max(distance, hnsw_distance),
                DbGroupingRecord.hash != hash,
            )
            .order_by(DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding))
            .limit(max(k, hnsw_candidates))
            .execution_options(**custom_options)
            .all()
        )

        reranked = self.rerank_candidates(candidates, embedding, distance, hash)
        return reranked[:k]

    @staticmethod
    def cosine_distance(
        embedding: np.ndarray,
        candidate_embedding: np.ndarray,
        embedding_norm: np.floating[Any] | None = None,
    ) -> float:
        if embedding_norm is None:
            embedding_norm = np.linalg.norm(embedding)

        candidate_norm = np.linalg.norm(candidate_embedding)
        dot_product = np.dot(candidate_embedding, embedding)
        return 1 - dot_product / (candidate_norm * embedding_norm)

    @sentry_sdk.tracing.trace
    def rerank_candidates(
        self, candidates: List[DbGroupingRecord], embedding: np.ndarray, distance: float, hash: str
    ) -> List[tuple[DbGroupingRecord, float]]:
        embedding_norm = np.linalg.norm(embedding)

        reranked = []
        for candidate in candidates:
            cos_distance = self.cosine_distance(
                embedding, candidate.stacktrace_embedding, embedding_norm
            )
            if cos_distance <= distance:
                reranked.append((candidate, cos_distance))

        reranked = sorted(reranked, key=lambda x: x[1])

        if candidates and reranked and candidates[0].hash != reranked[0][0].hash:
            span = sentry_sdk.Hub.current.scope.span
            if span:
                span.set_data("reranking_changed_output", 1)
                span.set_data("event_hash", hash)
                span.set_data("original_hash", candidates[0].hash)
                span.set_data("new_hash", reranked[0][0].hash)

            logger.info(
                "Reranking changed output: event_hash=%s, original_hash=%s, original_distance=%.4f, new_hash=%s, new_distance=%.4f",
                hash,
                candidates[0].hash,
                self.cosine_distance(embedding, candidates[0].stacktrace_embedding, embedding_norm),
                reranked[0][0].hash,
                reranked[0][1],
            )

        return reranked

    @sentry_sdk.tracing.trace
    def get_nearest_neighbors(self, issue: GroupingRequest) -> SimilarityResponse:
        """
        Retrieves the k nearest neighbors for a stacktrace within the same project and determines if they should be grouped.
        If no records should be grouped, inserts the request as a new GroupingRecord into the database.

        :param issue: The issue containing the stacktrace, similarity threshold, and number of nearest neighbors to find (k).
        :return: A SimilarityResponse object containing a list of GroupingResponse objects with the nearest group IDs,
                 stacktrace similarity scores, and grouping flags.
        """
        with Session() as session:
            embedding = self.encode_text(issue.stacktrace).astype("float32")

            results = self.query_nearest_k_neighbors(
                session,
                embedding,
                issue.project_id,
                issue.hash,
                NN_SIMILARITY_DISTANCE if issue.read_only else issue.threshold,
                issue.k,
                issue.hnsw_candidates,
                issue.hnsw_distance,
                issue.use_reranking,
            )

            # If no existing groups within the threshold, insert the request as a new GroupingRecord
            if not (issue.read_only or any(distance <= issue.threshold for _, distance in results)):
                logger.info(
                    "insert_new_grouping_record",
                    extra={
                        "input_hash": issue.hash,
                        "project_id": issue.project_id,
                        "stacktrace_length": len(issue.stacktrace),
                    },
                )
                self.insert_new_grouping_record(issue, embedding)

        similarity_response = SimilarityResponse(responses=[])
        for record, distance in results:
            should_group = distance <= issue.threshold

            if should_group:
                logger.info(
                    "should_group",
                    extra={
                        "input_hash": issue.hash,
                        "stacktrace_length": len(issue.stacktrace),
                        "parent_hash": record.hash,
                        "project_id": issue.project_id,
                    },
                )

            similarity_response.responses.append(
                GroupingResponse(
                    parent_hash=record.hash,
                    stacktrace_distance=distance,
                    should_group=should_group,
                )
            )

        return similarity_response

    @sentry_sdk.tracing.trace
    def bulk_create_and_insert_grouping_records(
        self, data: CreateGroupingRecordsRequest
    ) -> BulkCreateGroupingRecordsResponse:
        """
        Calls functions to create grouping records and bulk insert them.
        """
        if len(data.data) != len(data.stacktrace_list):
            return BulkCreateGroupingRecordsResponse(success=False, groups_with_neighbor={})

        groups_with_neighbor = self.insert_batch_grouping_records(data)
        return BulkCreateGroupingRecordsResponse(
            success=True, groups_with_neighbor=groups_with_neighbor
        )

    @sentry_sdk.tracing.trace
    def insert_batch_grouping_records(
        self, data: CreateGroupingRecordsRequest
    ) -> dict[str, GroupingResponse]:
        """
        Creates stacktrace embeddings in bulk.
        Checks if record has nearest neighbor or if it's already inserted, and inserts it if not.
        Returns group ids with neighbors.
        """
        groups_with_neighbor: dict[str, GroupingResponse] = {}
        embeddings = self.encode_multiple_texts(
            data.stacktrace_list, data.encode_stacktrace_batch_size
        )
        with Session() as session:
            for i, entry in enumerate(data.data):
                with sentry_sdk.start_span(
                    op="seer.grouping", description="insert single grouping record"
                ) as span:
                    span.set_data("stacktrace_len", len(data.stacktrace_list[i]))
                    embedding = embeddings[i].astype("float32")
                    nearest_neighbor = self.query_nearest_k_neighbors(
                        session,
                        embedding,
                        entry.project_id,
                        entry.hash,
                        data.threshold,
                        data.k,
                        data.hnsw_candidates,
                        data.hnsw_distance,
                        data.use_reranking,
                    )

                    if nearest_neighbor:
                        neighbor, distance = nearest_neighbor[0][0], nearest_neighbor[0][1]
                        response = GroupingResponse(
                            parent_hash=neighbor.hash,
                            stacktrace_distance=distance,
                            should_group=True,
                        )
                        groups_with_neighbor[str(entry.group_id)] = response
                    else:
                        insert_stmt = insert(DbGroupingRecord).values(
                            project_id=entry.project_id,
                            error_type=entry.exception_type,
                            hash=entry.hash,
                            stacktrace_embedding=embedding,
                        )

                        session.execute(
                            insert_stmt.on_conflict_do_nothing(
                                index_elements=(DbGroupingRecord.project_id, DbGroupingRecord.hash)
                            )
                        )
                        session.commit()

        return groups_with_neighbor

    @sentry_sdk.tracing.trace
    def insert_new_grouping_record(self, issue: GroupingRequest, embedding: np.ndarray) -> None:
        """
        Inserts a new GroupingRecord into the database if the group_hash does not already exist.

        :param session: The database session.
        :param issue: The issue to insert as a new GroupingRecord.
        :param embedding: The embedding of the stacktrace.
        """
        with Session() as session:
            insert_stmt = insert(DbGroupingRecord).values(
                project_id=issue.project_id,
                stacktrace_embedding=embedding,
                hash=issue.hash,
                error_type=issue.exception_type,
            )

            try:
                session.execute(
                    insert_stmt.on_conflict_do_nothing(
                        index_elements=(DbGroupingRecord.project_id, DbGroupingRecord.hash)
                    )
                )
                session.commit()
            except IntegrityError:
                logger.exception(
                    "group_already_exists_in_seer_db",
                    extra={
                        "project_id": issue.project_id,
                        "stacktrace_length": len(issue.stacktrace),
                        "input_hash": issue.hash,
                    },
                )

    @sentry_sdk.tracing.trace
    def delete_grouping_records_for_project(self, project_id: int) -> bool:
        """
        Deletes grouping records for a project.
        """
        with Session() as session:
            session.query(DbGroupingRecord).filter_by(project_id=project_id).delete()
            session.commit()
        return True

    @sentry_sdk.tracing.trace
    def delete_grouping_records_by_hash(
        self, data: DeleteGroupingRecordsByHashRequest
    ) -> DeleteGroupingRecordsByHashResponse:
        """
        Deletes grouping records that match a list of hashes.
        """
        with Session() as session:
            session.query(DbGroupingRecord).filter(
                DbGroupingRecord.project_id == data.project_id,
                DbGroupingRecord.hash.in_(data.hash_list),
            ).delete()
            session.commit()
        return DeleteGroupingRecordsByHashResponse(success=True)
