import difflib
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ValidationInfo, field_validator
from sentence_transformers import SentenceTransformer

from seer.db import DbGroupingRecord, Session


class GroupingRequest(BaseModel):
    group_id: int
    project_id: int
    stacktrace: str
    message: str
    k: int = 1
    threshold: float = 0.01

    @field_validator("stacktrace", "message")
    @classmethod
    def check_field_is_not_empty(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError(f"{info.field_name} must be provided and not empty.")
        return v


class GroupingRecord(BaseModel):
    group_id: int
    project_id: int
    message: str
    stacktrace_embedding: np.ndarray

    def to_db_model(self) -> DbGroupingRecord:
        return DbGroupingRecord(
            group_id=self.group_id,
            project_id=self.project_id,
            message=self.message,
            stacktrace_embedding=self.stacktrace_embedding,
        )


class GroupingResponse(BaseModel):
    parent_group_id: Optional[int]
    stacktrace_similarity: float
    message_similarity: float
    should_group: bool


class SimilarityResponse(BaseModel):
    responses: List[GroupingResponse]


class GroupingLookup:
    """Manages the grouping of similar stack traces using sentence embeddings and pgvector for similarity search.

    Attributes:
        model (SentenceTransformer): The sentence transformer model for encoding text.

    """

    def __init__(self, model_path: str):
        """
        Initializes the GroupingLookup with the sentence transformer model.

        :param model_path: Path to the sentence transformer model.
        """
        self.model = SentenceTransformer(model_path, trust_remote_code=True)

    def encode_text(self, stacktrace: str) -> np.ndarray:
        """
        Encodes the stacktrace using the sentence transformer model.

        :param stacktrace: The stacktrace to encode.
        :return: The embedding of the stacktrace.
        """
        return self.model.encode(stacktrace)

    def get_nearest_neighbors(self, issue: GroupingRequest) -> SimilarityResponse:
        """
        Retrieves the k nearest neighbors for a stacktrace within the same project and determines if they should be grouped.
        If no records should be grouped, inserts the request as a new GroupingRecord into the database.

        :param issue: The issue containing the stacktrace, similarity threshold, and number of nearest neighbors to find (k).
        :return: A SimilarityResponse object containing a list of GroupingResponse objects with the nearest group IDs,
                 stacktrace similarity scores, message similarity scores, and grouping flags.
        """
        embedding = self.encode_text(issue.stacktrace).astype("float32")
        with Session() as session:
            results = (
                session.query(
                    DbGroupingRecord,
                    DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding).label(
                        "distance"
                    ),
                )
                .filter(
                    DbGroupingRecord.project_id == issue.project_id,
                    DbGroupingRecord.stacktrace_embedding.cosine_distance(embedding) <= 0.15,
                    DbGroupingRecord.group_id != issue.group_id,
                )
                .order_by(DbGroupingRecord.stacktrace_embedding.nearest(embedding))
                .limit(issue.k)
                .all()
            )

            similarity_response = SimilarityResponse(responses=[])
            should_group_flag = False
            for record in results:
                message_similarity_score = difflib.SequenceMatcher(
                    None, issue.message, record.message
                ).ratio()
                should_group = record.distance <= issue.threshold
                should_group_flag = should_group_flag or should_group

                similarity_response.responses.append(
                    GroupingResponse(
                        parent_group_id=record.group_id,
                        stacktrace_similarity=record.distance,
                        message_similarity=message_similarity_score,
                        should_group=should_group,
                    )
                )

            if not should_group_flag:
                self.insert_new_grouping_record(session, issue, embedding)

            session.commit()

        return similarity_response

    def insert_new_grouping_record(self, session, issue: GroupingRequest, embedding: np.ndarray):
        """
        Inserts a new GroupingRecord into the database.

        :param session: The database session.
        :param issue: The issue to insert as a new GroupingRecord.
        :param embedding: The embedding of the stacktrace.
        """
        new_record = GroupingRecord(
            group_id=issue.group_id,
            project_id=issue.project_id,
            message=issue.message,
            stacktrace_embedding=embedding,
        ).to_db_model()
        session.add(new_record)
