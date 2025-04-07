import numpy as np
import pytest
from vertexai.language_models import (  # type: ignore[import-untyped]
    TextEmbedding,
    TextEmbeddingInput,
)

from seer.automation.agent.embeddings import GoogleProviderEmbeddings


class TestGoogleProviderEmbeddings:
    @pytest.fixture
    def model(self):
        return GoogleProviderEmbeddings.model("text-embedding-005")

    @pytest.fixture(autouse=True)
    def patch_from_pretrained(self, monkeypatch: pytest.MonkeyPatch):
        # test_encode passes without this patch, i.e., when actually hitting Google.
        # But VCR doesn't seem to store the requests or responses, maybe b/c they're gRPC calls?

        class MockTextEmbeddingModel:
            def get_embeddings(self, texts: list[TextEmbeddingInput], *args, **kwargs):
                output_dimensionality = kwargs.get("output_dimensionality") or 768
                text_embeddings: list[TextEmbedding] = []
                for text_embedding_input in texts:
                    rng = np.random.default_rng(seed=abs(hash(text_embedding_input.text)))
                    embedding_unnormalized = rng.random(size=(output_dimensionality,))
                    embedding = embedding_unnormalized / np.linalg.norm(
                        embedding_unnormalized, axis=0, keepdims=True
                    )
                    text_embeddings.append(TextEmbedding(values=embedding))
                return text_embeddings

        monkeypatch.setattr(
            "vertexai.language_models._language_models.TextEmbeddingModel.from_pretrained",
            lambda model_name: MockTextEmbeddingModel(),
        )

    @pytest.mark.parametrize(
        "texts",
        (
            [
                "text 1",
                "text two",
                "text longer",
                "text too long for other batches",
                "back to",
                "short",
                "texts again",
                "oh wait there's another one here",
                "here at last",
            ],
        ),
    )
    @pytest.mark.parametrize("max_batch_size", (1, 2, 3))
    @pytest.mark.parametrize("max_tokens", (3, 10))
    @pytest.mark.parametrize("avg_num_chars_per_token", (4.0,))
    def test_prepare_batches(
        self,
        texts: list[str],
        max_batch_size: int,
        max_tokens: int,
        avg_num_chars_per_token: float,
        model: GoogleProviderEmbeddings,
    ):
        batches = []
        for batch in model._prepare_batches(
            texts,
            max_batch_size=max_batch_size,
            max_tokens=max_tokens,
            avg_num_chars_per_token=avg_num_chars_per_token,
        ):
            assert len(batch) <= max_batch_size
            if any((len(text) / avg_num_chars_per_token) > max_tokens for text in batch):
                assert len(batch) == 1
            else:
                num_tokens_batch_estimate = (
                    sum(len(text) for text in batch) / avg_num_chars_per_token
                )
                assert num_tokens_batch_estimate <= max_tokens
            batches.append(batch)

        flat = [text for batch in batches for text in batch]
        assert flat == texts

    @pytest.mark.parametrize("show_progress_bar", (True, False))
    def test_encode(self, model: GoogleProviderEmbeddings, show_progress_bar: bool):
        texts = [
            "text 1",
            "text 2",
            "text 2",
            "text 1",
            "text 3",
            "text 4",
            "text 1",
        ]
        embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        for text, embedding in zip(texts, embeddings, strict=True):
            embedding_expected = model.encode(text)
            assert np.allclose(embedding, embedding_expected, atol=1e-4)
