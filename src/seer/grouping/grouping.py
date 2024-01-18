import difflib
import pickle
from typing import Any, List, Optional

import faiss  # type: ignore
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator, validator
from sentence_transformers import SentenceTransformer


class GroupingRequest(BaseModel):
    group_id: int
    stacktrace: str
    message: str
    k: int = 1
    threshold: float = 0.99

    @field_validator("stacktrace", "message")
    @classmethod
    def check_field_is_not_empty(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError(f"{info.field_name} must be provided and not empty.")
        return v


class GroupingRecord(BaseModel):
    group_id: int
    stacktrace: str
    message: str
    embeddings: Any

    @validator("embeddings", pre=True, allow_reuse=True)
    def check_embeddings(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Embeddings must be a numpy array.")
        return v


class GroupingResponse(BaseModel):
    parent_group_id: Optional[int]
    stacktrace_similarity: float
    message_similarity: float
    should_group: bool


class SimilarityResponse(BaseModel):
    responses: List[GroupingResponse]


class GroupingLookup:
    """
    Manages the grouping of similar stack traces using sentence embeddings.

    Attributes:
        model (SentenceTransformer): The sentence transformer model for encoding text.
        data (pd.DataFrame): The dataset containing stacktrace embeddings.
        index (faiss.IndexFlat): The FAISS index for similarity search.
    """

    def __init__(self, model_path: str, data_path: str):
        """
        Initializes the GroupingLookup with the model and preprocessed data. Generates
        faiss index for similarity search.

        Args:
            model_path (str): Path to the sentence transformer model.
            data_path (str): Path to the preprocessed data with stacktrace embeddings.
        """
        self.model = SentenceTransformer(model_path)
        with open(data_path, "rb") as file:
            self.data = pickle.load(file)
        embeddings = np.vstack(self.data["embeddings"].values).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def encode_text(self, stacktrace: str) -> np.ndarray:
        """
        Encodes a stacktrace into an embedding and normalizes the result.

        Args:
            stacktrace (str): The stacktrace to encode.

        Returns:
            np.ndarray: The normalized embedding of the stacktrace.
        """
        embedding = self.model.encode(stacktrace)
        embedding /= np.linalg.norm(embedding)
        return embedding

    def add_new_record_to_index(self, new_record: GroupingRecord) -> None:
        """
        Adds a new stacktrace embedding to the index and updates the dataset.

        Args:
            new_record (GroupingRecord): The new stacktrace record to add.

        Returns:
            None
        """
        self.data = pd.concat([self.data, pd.DataFrame([new_record.dict()])], ignore_index=True)
        self.index.add(np.array([new_record.embeddings], dtype="float32"))

    def get_nearest_neighbors(self, issue: GroupingRequest) -> SimilarityResponse:
        """
        Retrieves the k nearest neighbors for a stacktrace and determines if they should be grouped,
        ensuring that an issue is not grouped with itself.

        Args:
            issue (GroupingRequest): The issue containing the stacktrace, similarity threshold, and
                                     number of nearest neighbors to find (k)

        Returns:
            SimilarityResponse: A SimilarityResponse object containing a list of GroupingResponse objects with the nearest group IDs,
                                stacktrace similarity scores, message similarity scores, and grouping flags.
        """
        embedding = self.encode_text(issue.stacktrace).astype("float32")
        embedding = np.expand_dims(embedding, axis=0)
        # Find one extra neighbor to account for the issue itself
        distances, indices = self.index.search(embedding, k=issue.k + 1)
        similarity_response = SimilarityResponse(responses=[])

        for i in range(indices.shape[1]):
            group_id = self.data.iloc[indices[0][i]]["group_id"]
            if group_id == issue.group_id:
                continue  # Skip if the found group is the same as the issue's group

            stacktrace_similarity_score = distances[0][i]
            neighboring_message = self.data.iloc[indices[0][i]]["message"]
            message_similarity_score = difflib.SequenceMatcher(
                None, issue.message, neighboring_message
            ).ratio()
            should_group = stacktrace_similarity_score >= issue.threshold
            if not should_group:
                new_record = GroupingRecord(
                    group_id=issue.group_id,
                    embeddings=np.squeeze(embedding),
                    message=issue.message,
                    stacktrace=issue.stacktrace,
                )
                self.add_new_record_to_index(new_record)
                parent_group_id = None
            else:
                parent_group_id = group_id

            similarity_response.responses.append(
                GroupingResponse(
                    parent_group_id=parent_group_id,
                    stacktrace_similarity=stacktrace_similarity_score,
                    message_similarity=message_similarity_score,
                    should_group=should_group,
                )
            )

            if len(similarity_response.responses) == issue.k:
                break

        return similarity_response
