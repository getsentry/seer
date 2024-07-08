import functools
import logging
import os

import billiard  # type: ignore[import-untyped]
import torch
from sentence_transformers import SentenceTransformer

from seer.env import Environment
from seer.injector import inject, injected
from seer.rpc import DummyRpcClient, RpcClient, SentryRpcClient
from seer.stubs import DummySentenceTransformer, can_use_model_stubs

# ALERT: Using magic number four. This is temporary code that ensures that AutopFix uses all 4
# cuda devices available. This "4" should match the number of celery sub-processes configured in celeryworker.sh.
EXPECTED_CUDA_DEVICES = 4
logger = logging.getLogger(__name__)


class ConsentError(Exception):
    """Exception raised when consent is not granted for an operation."""

    pass


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
    if can_use_model_stubs():
        return DummySentenceTransformer(embedding_size=768)
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


def process_repo_provider(provider: str) -> str:
    if provider.startswith("integrations:"):
        return provider.split(":")[1]
    return provider


@inject
def check_genai_consent(org_id: int, env: Environment = injected) -> bool:
    if env.NO_SENTRY_INTEGRATION:
        # If we are running in a local environment, we just pass this check
        return True

    response = get_sentry_client().call("get_organization_autofix_consent", org_id=org_id)

    if response and response.get("consent", False) is True:
        return True
    return False


def raise_if_no_genai_consent(org_id: int) -> None:
    if not check_genai_consent(org_id):
        raise ConsentError(f"Organization {org_id} has not consented to use GenAI")


@inject
def get_sentry_client(env: Environment) -> RpcClient:
    if env.NO_SENTRY_INTEGRATION == "1":
        rpc_client: DummyRpcClient = DummyRpcClient()
        rpc_client.dry_run = True
        return rpc_client
    else:
        return SentryRpcClient()
