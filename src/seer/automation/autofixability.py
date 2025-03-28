import logging
from dataclasses import dataclass

import torch
from scipy.special import softmax
from sentence_transformers import SentenceTransformer

from seer.stubs import DummySentenceTransformer, can_use_model_stubs

logger = logging.getLogger(__name__)


def _load_model(model_path: str) -> SentenceTransformer:
    if can_use_model_stubs():
        return DummySentenceTransformer(embedding_size=384)

    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading transformer model to device {model_device}")
    return SentenceTransformer(model_path, device=model_device)


@dataclass
class AutofixabilityModel:
    model_path: str

    def __post_init__(self):
        self.model = _load_model(self.model_path)
        self.fixable_range = [
            "This issue is complex and very difficult to resolve",
            "This issue is in the codebase, simple and easily resolved",
        ]
        self.embeddings_fixable = self.model.encode(self.fixable_range)

    def score(self, issue_summary_input: str) -> float:
        embedding_issue_summary = self.model.encode(issue_summary_input)
        pred_probs = softmax(embedding_issue_summary @ self.embeddings_fixable.T, axis=-1)
        return float(pred_probs[1])  # upcast from np.float32 to float
