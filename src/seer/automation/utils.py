import functools
import logging
import os
from typing import Any

import billiard  # type: ignore[import-untyped]
import torch
from sentence_transformers import SentenceTransformer

# ALERT: Using magic number four. This is temporary code that ensures that AutopFix uses all 4
# cuda devices available. This "4" should match the number of celery sub-processes configured in celeryworker.sh.
EXPECTED_CUDA_DEVICES = 4
logger = logging.getLogger("autofix")
automation_logger = logging.getLogger("automation")


def _use_cuda():
    return os.getenv("USE_CUDA", "false").lower() in ("true", "t", "1")


def _get_torch_device_name():
    device: str = "cpu"
    cuda = _use_cuda()
    logger.debug(f"env USE_CUDA set to: {cuda}")
    if cuda and torch.cuda.is_available():
        if torch.cuda.device_count() >= EXPECTED_CUDA_DEVICES:
            try:
                index: int = billiard.process.current_process().index
                if index < 0 or index >= EXPECTED_CUDA_DEVICES:
                    logger.warn(
                        f"CUDA device selection: invalid process index {index}. Defaulting to active GPU."
                    )
                    device = "cuda"
                else:
                    device = f"cuda:{index}"
            except Exception as e:
                logger.warn(
                    "CUDA device selection: unable to look up celery process index. Defaulting to active GPU.",
                    exc_info=e,
                )
                device = "cuda"
        else:
            logger.info(
                f"CUDA device selection: found {torch.cuda.device_count()} CUDA devices which is less than the required {EXPECTED_CUDA_DEVICES}. Defaulting to active GPU"
            )
            device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Using {device} device for embedding model")
    return device


@functools.cache
def _get_embedding_model_on(device_name: str):
    device = torch.device(device_name)
    model = SentenceTransformer(
        os.path.join("./", "models", "autofix_embeddings_v0"),
        trust_remote_code=True,
    ).to(device=device)
    model.max_seq_length = 4096
    return model


def get_embedding_model():
    return _get_embedding_model_on(_get_torch_device_name())


def make_done_signal(id: str | int) -> str:
    return f"done:{id}"
