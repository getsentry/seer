import numpy as np
import pandas as pd
import torch
from joblib import load
from sentence_transformers import SentenceTransformer


class SeverityInference:
    def __init__(self, embeddings_path, classifier_path):
        """Initialize the inference class with pre-trained models and tokenizer."""
        self.embeddings_model = SentenceTransformer(
            embeddings_path,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )
        self.classifier = load(classifier_path)

    def get_embeddings(self, text):
        """Generate embeddings for the given text using the pre-trained model."""
        return self.embeddings_model.encode(text, convert_to_numpy=True)

    def severity_score(self, data):
        """Predict the severity score for the given text using the pre-trained classifier."""
        embeddings = self.get_embeddings(data.get("message")).reshape(-1)
        has_stacktrace = data.get("has_stacktrace", 0)

        handled = data.get("handled")
        handled_true = 1 if handled is True else 0
        handled_false = 1 if handled is False else 0
        handled_unknown = 1 if handled is None else 0

        input_data = np.append(
            embeddings.reshape(1, -1),
            [[has_stacktrace, handled_true, handled_false, handled_unknown]],
            axis=1,
        )

        pred = self.classifier.predict_proba(input_data)[0][1]
        return round(min(1.0, max(0.0, pred)), 2)
