import difflib
import logging
from typing import List, Optional

import numpy as np
import sentry_sdk
import sqlalchemy.orm
import torch
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from sentence_transformers import SentenceTransformer
from sentry_sdk import metrics
from sqlalchemy.dialects.postgresql import insert

from seer.db import DbGroupingRecord, Session
from seer.stubs import DummySentenceTransformer, can_use_model_stubs

logger = logging.getLogger("grouping")

NN_GROUPING_DISTANCE = 0.01
NN_SIMILARITY_DISTANCE = 0.05


class GroupingRequest(BaseModel):
    project_id: int
    stacktrace: str
    message: str
    hash: str
    exception_type: Optional[str] = None
    k: int = 1
    threshold: float = NN_GROUPING_DISTANCE
    read_only: bool = False

    @field_validator("stacktrace", "message")
    @classmethod
    def check_field_is_not_empty(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError(f"{info.field_name} must be provided and not empty.")
        return v


class GroupingResponse(BaseModel):
    parent_hash: str
    stacktrace_distance: float
    message_distance: float
    should_group: bool


class SimilarityResponse(BaseModel):
    responses: List[GroupingResponse] = Field(default_factory=list)

    # This will happen naturally -- retries on inflight messages, halting problems related to serialization
    # in processing queues, etc, etc, etc.  Do not assume that duplicates are necessarily a bug to be fixed,
    # they are the natural and healthy consequence of trade offs in system design.
    # This value is mostly for metrics and awareness, it does not indicate failure.
    had_duplicate: bool = False


class SimilarityBenchmarkResponse(BaseModel):
    embedding: List[float]


class CreateGroupingRecordData(BaseModel):
    group_id: int
    hash: str
    project_id: int
    message: str
    exception_type: Optional[str] = None


class CreateGroupingRecordsRequest(BaseModel):
    data: List[CreateGroupingRecordData]
    stacktrace_list: List[str]
    encode_stacktrace_batch_size: int = 1


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
    message: str
    stacktrace_embedding: np.ndarray
    hash: str
    error_type: Optional[str] = None

    def to_db_model(self) -> DbGroupingRecord:
        return DbGroupingRecord(
            project_id=self.project_id,
            message=self.message,
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
    def encode_text(self, stacktrace: str) -> np.ndarray:
        """
        Encodes the stacktrace using the sentence transformer model.

        :param stacktrace: The stacktrace to encode.
        :return: The embedding of the stacktrace.
        """
        return self.model.encode(stacktrace)

    @sentry_sdk.tracing.trace
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
    ) -> List[tuple[DbGroupingRecord, float]]:
        cos_dist = DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding).label(
            "distance"
        )
        return (
            session.query(
                DbGroupingRecord,
                cos_dist,
            )
            .filter(
                DbGroupingRecord.project_id == project_id,
                cos_dist <= distance,
                DbGroupingRecord.hash != hash,
            )
            .order_by("distance")
            .limit(k)
            .all()
        )

    @sentry_sdk.tracing.trace
    def get_nearest_neighbors(self, issue: GroupingRequest) -> SimilarityResponse:
        """
        Retrieves the k nearest neighbors for a stacktrace within the same project and determines if they should be grouped.
        If no records should be grouped, inserts the request as a new GroupingRecord into the database.

        :param issue: The issue containing the stacktrace, similarity threshold, and number of nearest neighbors to find (k).
        :return: A SimilarityResponse object containing a list of GroupingResponse objects with the nearest group IDs,
                 stacktrace similarity scores, message similarity scores, and grouping flags.
        """
        responses = self.bulk_nearest_neighbor([issue])
        if responses:
            return responses[0]

        raise AssertionError("bulk_nearest_neighbor() didn't return a response!")

    @sentry_sdk.tracing.trace
    def bulk_create_and_insert_grouping_records(
        self, data: CreateGroupingRecordsRequest
    ) -> BulkCreateGroupingRecordsResponse:
        """
        Calls functions to create grouping records and bulk insert them.
        """
        if len(data.data) != len(data.stacktrace_list):
            return BulkCreateGroupingRecordsResponse(success=False, groups_with_neighbor={})

        responses: list[SimilarityResponse] = self.bulk_nearest_neighbor(
            [
                GroupingRequest(
                    project_id=create_group.project_id,
                    stacktrace=stacktrace,
                    message=create_group.message,
                    hash=create_group.hash,
                    exception_type=create_group.exception_type,
                    k=1,
                )
                for create_group, stacktrace in zip(data.data, data.stacktrace_list)
            ],
            data.encode_stacktrace_batch_size,
        )

        return BulkCreateGroupingRecordsResponse(
            success=True,
            groups_with_neighbor={
                str(request.group_id): response.responses[0]
                for request, response in zip(data.data, responses)
                if response.responses
            },
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
                embedding = embeddings[i].astype("float32")
                nearest_neighbor = self.query_nearest_k_neighbors(
                    session,
                    embedding,
                    entry.project_id,
                    entry.hash,
                    NN_GROUPING_DISTANCE,
                    1,
                )

                if nearest_neighbor:
                    neighbor, distance = nearest_neighbor[0][0], nearest_neighbor[0][1]
                    message_similarity_score = difflib.SequenceMatcher(
                        None, entry.message, neighbor.message
                    ).ratio()
                    response = GroupingResponse(
                        parent_hash=neighbor.hash,
                        stacktrace_distance=distance,
                        message_distance=1.0 - message_similarity_score,
                        should_group=True,
                    )
                    groups_with_neighbor[str(entry.group_id)] = response
                else:
                    insert_stmt = insert(DbGroupingRecord).values(
                        project_id=entry.project_id,
                        message=entry.message,
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
    def bulk_nearest_neighbor(
        self,
        requests: List[GroupingRequest],
        encode_stacktrace_batch_size: int = 1,
    ) -> list[SimilarityResponse]:
        """
        Given the grouping requests, encodes and searches for nearest neighbors according
        to that request.  When that request is not read_only and no neighbors within the threshold
        is found, a new one is created.  When duplicates are encountered, that is noted in the response.
        All neighbors within the given threshold are returned.

        Responses are always in order of the requests, and of the same size.
        """
        result: list[SimilarityResponse] = []
        embeddings = self.encode_multiple_texts(
            [r.stacktrace for r in requests], encode_stacktrace_batch_size
        )
        with Session() as session:
            request: GroupingRequest
            embedding: np.ndarray
            for embedding, request in zip(embeddings, requests):
                embedding = embedding.astype("float32")
                nearest_neighbors = self.query_nearest_k_neighbors(
                    session,
                    embedding,
                    request.project_id,
                    request.hash,
                    request.threshold if not request.read_only else NN_GROUPING_DISTANCE,
                    request.k,
                )

                threshold_matches = any(
                    distance <= request.threshold for _, distance in nearest_neighbors
                )
                response = SimilarityResponse()
                result.append(response)
                if not request.read_only and not threshold_matches:
                    response.had_duplicate = self.upsert_grouping_record(
                        embedding, request, session
                    )

                for record, distance in nearest_neighbors:
                    message_similarity_score = difflib.SequenceMatcher(
                        None, request.message, record.message
                    ).ratio()
                    response.responses.append(
                        GroupingResponse(
                            parent_hash=record.hash,
                            stacktrace_distance=distance,
                            message_distance=1.0 - message_similarity_score,
                            should_group=distance <= request.threshold,
                        )
                    )

            session.commit()
        return result

    def upsert_grouping_record(
        self, embedding: np.ndarray, request: GroupingRequest, session: sqlalchemy.orm.Session
    ) -> bool:
        insert_stmt = insert(DbGroupingRecord).values(
            project_id=request.project_id,
            message=request.message,
            error_type=request.exception_type,
            hash=request.hash,
            stacktrace_embedding=embedding,
        )
        row = session.execute(
            insert_stmt.on_conflict_do_nothing(
                index_elements=(DbGroupingRecord.project_id, DbGroupingRecord.hash)
            ).returning(DbGroupingRecord.id)
        )
        had_duplicate = row.first() is None

        if had_duplicate:
            # this isn't necessarily a huge deal unless unexpected increases occur not correlated with
            # volume.
            metrics.incr("seer.grouping.grouping_duplicate")

        return had_duplicate

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
