import typing
from typing import Optional

import numpy as np
import sentry_sdk
import torch
from joblib import load
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentry_services.seer.severity_adaptors import FromScoreRequestAdaptor, ToScoreResponseAdaptor
from sentry_services.seer.severity_pb2 import ScoreResponse


class SeverityRequest(BaseModel, FromScoreRequestAdaptor):
    message: str = ""
    has_stacktrace: int = 0
    handled: Optional[bool] = None
    trigger_timeout: bool | None = None
    trigger_error: bool | None = None

    def adapt_from_message(self, value: str):
        self.message = value

    def adapt_from_has_stacktrace(self, value: int):
        self.has_stacktrace = value

    def adapt_from_handled(self, value: bool):
        self.handled = value


class SeverityResponse(BaseModel, ToScoreResponseAdaptor):
    severity: float = 0.0

    def apply_to_severity(self, proto: ScoreResponse, val: typing.Optional[float] = None):
        super().apply_to_severity(proto, self.severity)


class SeverityInference:
    def __init__(self, embeddings_path, classifier_path):
        """Initialize the inference class with pre-trained models and tokenizer."""

        pass
        # def init_embeddings_model(path: str):
        #     """Initialize embeddings model."""
        #     embeddings_model = SentenceTransformer(
        #         path,
        #         device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
        #     )
        #     test_str = """Error: [GraphQL error]: Message: "Not a team repl", Location: [{"line":2,"column":3}], Path: ["startTeamReplPresenceSession"]..."""
        #     _ = embeddings_model.encode(test_str, convert_to_numpy=True)  # Ensure warm start
        #     return embeddings_model
        #
        # self.embeddings_model = init_embeddings_model(embeddings_path)
        # self.classifier = load(classifier_path)

    def get_embeddings(self, text) -> np.ndarray:
        """Generate embeddings for the given text using the pre-trained model."""
        return self.embeddings_model.encode(text, convert_to_numpy=True)

    def severity_score(self, data: SeverityRequest) -> SeverityResponse:
        """Predict the severity score for the given text using the pre-trained classifier."""
        return SeverityResponse(severity=0.6)
        #
        # with sentry_sdk.start_span(op="severity.embeddings"):
        #     embeddings = self.get_embeddings(data.message).reshape(-1)
        #
        # with sentry_sdk.start_span(op="severity.classification"):
        #     has_stacktrace = data.has_stacktrace
        #
        #     handled = data.handled
        #     handled_true = 1 if handled is True else 0
        #     handled_false = 1 if handled is False else 0
        #     handled_unknown = 1 if handled is None else 0
        #
        #     input_data = np.append(
        #         embeddings.reshape(1, -1),
        #         [[has_stacktrace, handled_true, handled_false, handled_unknown]],
        #         axis=1,
        #     )
        #
        #     pred = self.classifier.predict_proba(input_data)[0][1]
        #
        # return SeverityResponse(severity=round(min(1.0, max(0.0, pred)), 2))
