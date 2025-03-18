from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
from more_itertools import chunked
from vertexai.language_models import (  # type: ignore[import-untyped]
    TextEmbeddingInput,
    TextEmbeddingModel,
)

from seer.automation.agent.client import GeminiProvider
from seer.automation.utils import batch_texts_by_token_count
from seer.utils import backoff_on_exception


@dataclass
class GoogleProviderEmbeddings:
    model_name: str
    provider_name = "Google"

    task_type: str | None = None
    """
    [More info on task types](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types).
    """
    output_dimensionality: int | None = None

    def get_client(self) -> TextEmbeddingModel:
        model = TextEmbeddingModel.from_pretrained(self.model_name)
        # Couldn't find built-in retry. Add in case it's missing.
        retrier = backoff_on_exception(
            GeminiProvider.is_completion_exception_retryable, max_tries=4
        )
        model.get_embeddings = retrier(model.get_embeddings)
        return model

    @classmethod
    def model(
        cls, model_name: str, task_type: str | None = None, output_dimensionality: int | None = None
    ) -> "GoogleProviderEmbeddings":
        return cls(
            model_name=model_name, task_type=task_type, output_dimensionality=output_dimensionality
        )

    def _prepare_inputs(self, texts: Iterable[str]) -> list[TextEmbeddingInput]:
        return [TextEmbeddingInput(text, self.task_type) for text in texts]

    def _prepare_batches(
        self,
        texts: Iterable[str],
        max_batch_size: int,
        max_tokens: int,
        avg_num_chars_per_token: float = 4.0,  # https://ai.google.dev/gemini-api/docs/tokens?lang=python
    ):
        for batch in chunked(texts, n=max_batch_size):
            for subbatch in batch_texts_by_token_count(
                batch, max_tokens=max_tokens, avg_num_chars_per_token=avg_num_chars_per_token
            ):
                yield subbatch

    def encode(self, texts: list[str], auto_truncate: bool = True) -> npt.NDArray[np.float64]:
        """
        Returns embeddings with shape `(len(texts), output_dimensionality)`.
        Embeddings are already normalized.

        This method handles batching for you, and prevents duplicate texts from being encoded
        multiple times.

        By default, texts are truncated to 2048 tokens.
        Setting `auto_truncate=False` to disables truncation, but can result in API errors if a text exceeds this limit.
        """
        model = self.get_client()
        text_to_embedding: dict[str, list[float]] = {}
        texts_unique = list({text: None for text in texts})

        # https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#generative-ai-get-text-embedding-python_vertex_ai_sdk
        # - For each request, you're limited to 250 input texts in us-central1, and in other
        #   regions, the max input text is 5.
        # - The API has a maximum input token limit of 20,000
        for batch in self._prepare_batches(texts_unique, max_batch_size=5, max_tokens=20_000):
            text_embedding_inputs = self._prepare_inputs(batch)
            embeddings_batch = model.get_embeddings(
                text_embedding_inputs,
                auto_truncate=auto_truncate,
                output_dimensionality=self.output_dimensionality,
            )
            text_to_embedding.update(
                {
                    text: embedding.values
                    for text, embedding in zip(batch, embeddings_batch, strict=True)
                }
            )

        return np.array([text_to_embedding[text] for text in texts])


def cosine_similarity(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    dot_product = embeddings_a @ embeddings_b.T
    norm_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True).T
    return dot_product / (norm_a @ norm_b)
