import functools
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.cache
def get_embedding_model():
    model = SentenceTransformer(
        os.path.join("./", "models", "autofix_embeddings_v0"),
        trust_remote_code=True,
    ).to(get_torch_device())

    model.max_seq_length = 4096

    return model
