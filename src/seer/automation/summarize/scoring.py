import numpy as np
import numpy.typing as npt
from scipy.special import softmax


def embed_texts(texts: list[str], model: str = "text-embedding-3-large") -> npt.NDArray[np.float64]:
    """
    Embeds a list of texts as a 2-D array with shape `(len(texts), d)` via OpenAI API call.
    These embeddings are already L2-normalized.
    """
    from seer.automation.agent.client import OpenAiProvider

    client = OpenAiProvider.get_client()
    response = client.embeddings.create(input=texts, model=model)
    return np.array([data.embedding for data in response.data])


def predict_proba(
    embeddings_input: npt.NDArray[np.float64], embeddings_classes: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Zero-shot probability of each class.

    `embeddings_input` is an array with shape `(d,)` or `(num_obs, d)`.
    `embeddings_classes` is an array with shape `(num_classes, d)`.
    """
    similarities = embeddings_input @ embeddings_classes.T
    normalized: npt.NDArray[np.float64] = softmax(similarities, axis=-1)
    # Normalize to get the relative strengths
    return normalized


def cosine_similarity(
    embeddings1: npt.NDArray[np.float64], embeddings2: npt.NDArray[np.float64]
) -> np.float64 | npt.NDArray[np.float64]:
    """
    Compute the cosine similarity between two sets of embeddings.

    `embeddings1` and `embeddings2` are arrays, both with shape `(d,)` or `(num_obs, d)`.
    """
    return (embeddings1 * embeddings2).sum(axis=-1)
