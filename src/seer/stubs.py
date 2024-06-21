import hashlib
import os
from typing import List, Union

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor


class DummySentenceTransformer(SentenceTransformer):
    def __init__(self, embedding_size: int):
        super().__init__(None)  # No actual model needed
        self.embedding_size = embedding_size

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        embeddings_array: list[list[float]] = []
        is_single_result = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single_result = True

        for sentence in sentences:
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()
            bits: list[float] = []
            for byte in digest:
                for p in range(8):
                    bits.append(float((byte >> p) & 1))
                    if len(bits) >= self.embedding_size:
                        break
            bits.extend([0.0] * (self.embedding_size - len(bits)))
            embeddings_array.append(bits)

        embeddings = np.array(embeddings_array)
        if convert_to_tensor:
            embeddings = torch.tensor(embeddings)

        if is_single_result:
            return embeddings[0]
        return embeddings


def can_use_model_stubs():
    return os.environ.get("NO_REAL_MODELS", "").lower() in ("true", "t", "1")
