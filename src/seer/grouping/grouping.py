import difflib
import logging
from typing import List, Optional

import numpy as np
import sentry_sdk
import torch
from pydantic import BaseModel, ValidationInfo, field_validator
from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import IntegrityError

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
    error_type: Optional[str] = None
    k: int = 1
    threshold: float = NN_GROUPING_DISTANCE
    read_only: bool = False

    @field_validator("stacktrace", "message")
    @classmethod
    def check_field_is_not_empty(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError(f"{info.field_name} must be provided and not empty.")
        return v


class CreateGroupingRecordData(BaseModel):
    group_id: int
    hash: str
    project_id: int
    message: str


class CreateGroupingRecordsRequest(BaseModel):
    data: List[CreateGroupingRecordData]
    stacktrace_list: List[str]


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


class GroupingResponse(BaseModel):
    parent_hash: str
    stacktrace_distance: float
    message_distance: float
    should_group: bool


class SimilarityResponse(BaseModel):
    responses: List[GroupingResponse]


class BulkCreateGroupingRecordsResponse(BaseModel):
    success: bool
    groups_with_neighbor: dict[str, GroupingResponse]


class SimilarityBenchmarkResponse(BaseModel):
    embedding: List[float]


def _load_model(model_path: str) -> SentenceTransformer:
    if can_use_model_stubs():
        return DummySentenceTransformer(embedding_size=768)

    model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Loading transformer model to device {model_device}")
    SentenceTransformer(
        model_path,
        trust_remote_code=True,
        device=model_device,
    )
    sentry_sdk.capture_message(f"GroupingLookup model initialized using device: {model_device}")


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

    def encode_text(self, stacktrace: str) -> np.ndarray:
        """
        Encodes the stacktrace using the sentence transformer model.

        :param stacktrace: The stacktrace to encode.
        :return: The embedding of the stacktrace.
        """
        return self.model.encode(stacktrace)

    def encode_multiple_texts(self, stacktraces: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Encodes multiple stacktraces in batches using the sentence transformer model.
        :param stacktraces: The list of stacktraces to encode.
        :param batch_size: The batch size used for the computation.
        :return: The embeddings of the stacktraces.
        """

        return self.model.encode(sentences=stacktraces, batch_size=batch_size)

    def query_nearest_k_neighbors(
        self,
        session,
        embedding,
        project_id: int,
        hash: str,
        distance: float,
        k: int,
    ) -> List[tuple[DbGroupingRecord, float]]:
        return (
            session.query(
                DbGroupingRecord,
                DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding).label("distance"),
            )
            .filter(
                DbGroupingRecord.project_id == project_id,
                DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding) <= distance,
                DbGroupingRecord.hash != hash,
            )
            .order_by("distance")
            .limit(k)
            .all()
        )

    def get_nearest_neighbors(self, issue: GroupingRequest) -> SimilarityResponse:
        """
        Retrieves the k nearest neighbors for a stacktrace within the same project and determines if they should be grouped.
        If no records should be grouped, inserts the request as a new GroupingRecord into the database.

        :param issue: The issue containing the stacktrace, similarity threshold, and number of nearest neighbors to find (k).
        :return: A SimilarityResponse object containing a list of GroupingResponse objects with the nearest group IDs,
                 stacktrace similarity scores, message similarity scores, and grouping flags.
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
                self.insert_new_grouping_record(session, issue, embedding)
            session.commit()

        similarity_response = SimilarityResponse(responses=[])
        for record, distance in results:
            message_similarity_score = difflib.SequenceMatcher(
                None, issue.message, record.message
            ).ratio()
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
                    message_distance=1.0 - message_similarity_score,
                    should_group=should_group,
                )
            )

        return similarity_response

    def bulk_create_and_insert_grouping_records(
        self, data: CreateGroupingRecordsRequest
    ) -> BulkCreateGroupingRecordsResponse:
        """
        Calls functions to create grouping records and bulk insert them.
        """
        if len(data.data) != len(data.stacktrace_list):
            return BulkCreateGroupingRecordsResponse(success=False, groups_with_neighbor={})

        records, groups_with_neighbor = self.create_grouping_record_objects(data)
        self.bulk_insert_new_grouping_records(records)
        return BulkCreateGroupingRecordsResponse(
            success=True, groups_with_neighbor=groups_with_neighbor
        )

    def create_grouping_record_objects(
        self, data: CreateGroupingRecordsRequest
    ) -> tuple[List[DbGroupingRecord], dict[str, GroupingResponse]]:
        """
        Creates stacktrace emebddings and record objects for the given data.
        Returns a list of created records.
        """
        records, groups_with_neighbor = [], {}
        embeddings = self.encode_multiple_texts(data.stacktrace_list)
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
                if not any(distance <= NN_GROUPING_DISTANCE for _, distance in nearest_neighbor):
                    logger.info(
                        "inserting a new grouping record in bulk",
                        extra={
                            "input_hash": entry.hash,
                            "stacktrace_length": len(data.stacktrace_list[i]),
                            "project_id": entry.project_id,
                        },
                    )

                    new_record = GroupingRecord(
                        hash=entry.hash,
                        project_id=entry.project_id,
                        message=entry.message,
                        stacktrace_embedding=embedding,
                    ).to_db_model()
                    records.append(new_record)
                else:
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

            return (records, groups_with_neighbor)

    def insert_new_grouping_record(
        self, session, issue: GroupingRequest, embedding: np.ndarray
    ) -> None:
        """
        Inserts a new GroupingRecord into the database if the group_hash does not already exist.
        If new grouping record was created, return the id.

        :param session: The database session.
        :param issue: The issue to insert as a new GroupingRecord.
        :param embedding: The embedding of the stacktrace.
        """
        existing_record = (
            session.query(DbGroupingRecord)
            .filter_by(hash=issue.hash, project_id=issue.project_id)
            .first()
        )

        if existing_record is None:
            new_record = GroupingRecord(
                project_id=issue.project_id,
                message=issue.message,
                stacktrace_embedding=embedding,
                hash=issue.hash,
            ).to_db_model()
            session.add(new_record)
        else:
            logger.info(
                "group_already_exists_in_seer_db",
                extra={
                    "existing_hash": existing_record.hash,
                    "project_id": issue.project_id,
                    "stacktrace_length": len(issue.stacktrace),
                    "input_hash": issue.hash,
                },
            )

    def bulk_insert_new_grouping_records(self, records: List[DbGroupingRecord]):
        """
        Bulk inserts new GroupingRecord into the database.
        :param records: List of records to be inserted
        """
        try:
            with Session() as session:
                session.bulk_save_objects(records)
                session.commit()
        except IntegrityError:
            with Session() as session:
                logger.info("Error in bulk insert. Attempting to insert records individually...")

                existing_records = (
                    session.query(DbGroupingRecord.hash, DbGroupingRecord.project_id)
                    .filter(
                        DbGroupingRecord.hash.in_([record.hash for record in records]),
                        DbGroupingRecord.project_id.in_([record.project_id for record in records]),
                    )
                    .all()
                )

                existing_records_set = {
                    (record.hash, record.project_id) for record in existing_records
                }

                records_to_insert = [
                    record
                    for record in records
                    if (record.hash, record.project_id) not in existing_records_set
                ]
                if not records_to_insert:
                    logger.info("No new records to insert.")
                    return
                session.bulk_save_objects(records_to_insert)
                session.commit()

    def delete_grouping_records_for_project(self, project_id: int) -> bool:
        """
        Deletes grouping records for a project.
        """
        with Session() as session:
            session.query(DbGroupingRecord).filter_by(project_id=project_id).delete()
            session.commit()
        return True
