import functools
import logging
import os
import re
from xml.etree import ElementTree as ET

import billiard  # type: ignore[import-untyped]
import torch
from sentence_transformers import SentenceTransformer

from seer.rpc import DummyRpcClient, RpcClient, SentryRpcClient
from seer.stubs import DummySentenceTransformer, can_use_model_stubs

# ALERT: Using magic number four. This is temporary code that ensures that AutopFix uses all 4
# cuda devices available. This "4" should match the number of celery sub-processes configured in celeryworker.sh.
EXPECTED_CUDA_DEVICES = 4
logger = logging.getLogger(__name__)


class ConsentError(Exception):
    """Exception raised when consent is not granted for an operation."""

    pass


class AgentError(Exception):
    """Exception to be ignored by the Sentry SDK and intended only for an AI agent to read"""

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


def check_genai_consent(org_id: int) -> bool:
    if os.environ.get("NO_SENTRY_INTEGRATION") == "1":
        # If we are running in a local environment, we just pass this check
        return True

    response = get_sentry_client().call("get_organization_autofix_consent", org_id=org_id)

    if response and response.get("consent", False) is True:
        return True
    return False


def raise_if_no_genai_consent(org_id: int) -> None:
    if not check_genai_consent(org_id):
        raise ConsentError(f"Organization {org_id} has not consented to use GenAI")


def get_sentry_client() -> RpcClient:
    if os.environ.get("NO_SENTRY_INTEGRATION") == "1":
        rpc_client: DummyRpcClient = DummyRpcClient()
        rpc_client.dry_run = True
        return rpc_client
    else:
        return SentryRpcClient()


def escape_xml_chars(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def unescape_xml_chars(s: str) -> str:
    return (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )


def escape_xml(s: str, tag: str) -> str:
    def replace_content(match):
        return match.group(0).replace(match.group(2), escape_xml_chars(match.group(2)))

    return re.sub(rf"<{tag}(\s+[^>]*)?>((.|\n)*?)</{tag}>", replace_content, s, flags=re.DOTALL)


def escape_multi_xml(s: str, tags: list[str]) -> str:
    for tag in tags:
        s = escape_xml(s, tag)

    return s


def remove_cdata(obj):
    """
    Recursively remove CDATA wrappings from any object.
    This is useful for objects created directly from XML that required CDATA to be well-formed.
    """

    def remove_cdata_from_string(s):
        cdata_pattern = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
        return cdata_pattern.sub(r"\1", s)

    if isinstance(obj, dict):
        return {key: remove_cdata(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [remove_cdata(item) for item in obj]
    elif isinstance(obj, str):
        return remove_cdata_from_string(obj)
    elif hasattr(obj, "__dict__"):  # Handle custom objects
        for attr in vars(obj):
            setattr(obj, attr, remove_cdata(getattr(obj, attr)))
        return obj
    else:
        return obj


def extract_text_inside_tags(content: str, tag: str, strip_newlines: bool = True) -> str:
    """
    Extract the text inside the specified XML tag.

    Args:
        content (str): The XML content.
        tag (str): The tag to extract the text from.

    Returns:
        str: The text inside the specified XML tag.
    """
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"

    start_index = content.find(start_tag)
    end_index = content.find(end_tag)

    if start_index == -1 or end_index == -1:
        return ""

    text = content[start_index + len(start_tag) : end_index]

    return text.strip("\n") if strip_newlines else text


def extract_xml_element_text(element: ET.Element, tag: str) -> str | None:
    """
    Extract the text from an XML element with the given tag.

    Args:
        element (ET.Element): The XML element to extract the text from.
        tag (str): The tag of the XML element to extract the text from.

    Returns:
        str: The text of the XML element with the given tag.
    """
    el = element.find(tag)

    if el is not None:
        return remove_cdata((el.text or "")).strip()

    return None
