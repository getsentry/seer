from typing import Optional, Protocol

import numpy as np
import sentry_sdk
import torch
from joblib import load
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier

from seer.stubs import DummySentenceTransformer, can_use_model_stubs


class SeverityRequest(BaseModel):
    message: str = ""
    has_stacktrace: int = 0
    handled: Optional[bool] = None
    trigger_timeout: bool | None = None
    trigger_error: bool | None = None


class SeverityResponse(BaseModel):
    severity: float = 0.0


class SeverityClassifier(Protocol):
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        pass


def _init_embeddings_model(path: str) -> SentenceTransformer:
    """Initialize embeddings model."""
    if can_use_model_stubs():
        return DummySentenceTransformer(embedding_size=384)

    embeddings_model = SentenceTransformer(
        path,
        device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
    )
    # Ensure warm start
    test_str = """Error: [GraphQL error]: Message: "Not a team repl", Location: [{"line":2,"column":3}], Path: ["startTeamReplPresenceSession"]..."""
    _ = embeddings_model.encode(test_str, convert_to_numpy=True, show_progress_bar=False)
    return embeddings_model


def _init_classifier(path: str) -> SeverityClassifier:
    if can_use_model_stubs():
        classifier = DummyClassifier()
        classifier.fit(np.array([0, 1, 1, -1]), np.array([1, 0, 0, 1]))
        return classifier

    return load(path)


class SeverityInference:
    embeddings_model: SentenceTransformer
    classifier: SeverityClassifier

    def __init__(self, embeddings_path: str, classifier_path: str):
        """Initialize the inference class with pre-trained models and tokenizer."""

        self.embeddings_model = _init_embeddings_model(embeddings_path)
        self.classifier = _init_classifier(classifier_path)

    def get_embeddings(self, text) -> np.ndarray:
        """Generate embeddings for the given text using the pre-trained model."""
        return self.embeddings_model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def severity_score(self, data: SeverityRequest) -> SeverityResponse:
        """Predict the severity score for the given text using the pre-trained classifier."""
        with sentry_sdk.start_span(op="severity.embeddings"):
            embeddings = self.get_embeddings(data.message).reshape(-1)

        with sentry_sdk.start_span(op="severity.classification"):
            has_stacktrace = data.has_stacktrace

            handled = data.handled
            handled_true = 1 if handled is True else 0
            handled_false = 1 if handled is False else 0
            handled_unknown = 1 if handled is None else 0

            input_data = np.append(
                embeddings.reshape(1, -1),
                [[has_stacktrace, handled_true, handled_false, handled_unknown]],
                axis=1,
            )

            pred = self.classifier.predict_proba(input_data)[0][1]

        return SeverityResponse(severity=round(min(1.0, max(0.0, pred)), 2))
