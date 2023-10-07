import sentry_sdk
import torch
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from joblib import load


class SeverityInference:
    def __init__(self, embeddings_path, tokenizer_path, classifier_path):
        """Initialize the inference class with pre-trained models and tokenizer."""
        self.embeddings_model = RobertaForSequenceClassification.from_pretrained(
            embeddings_path
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.classifier = load(classifier_path)

    def get_embeddings(self, text, max_len=128):
        """Generate embeddings for the given text using the pre-trained model."""
        # Tokenize the input string
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Move model to the specified device and set it to eval mode
        model = self.embeddings_model.to(self.device)
        model.eval()

        # Forward pass
        with torch.no_grad():
            outputs = model.roberta(input_ids, attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze()

        return embeddings.cpu().numpy()

    def severity_score(self, data):
        """Predict the severity score for the given text using the pre-trained classifier."""
        with sentry_sdk.start_span(op="model.severity", description="get_embeddings"):
            embeddings = self.get_embeddings(data.get("message"))
        with sentry_sdk.start_span(op="model.severity", description="predict_proba"):
            has_stacktrace = data.get("has_stacktrace", 0)
            log_level_error = (
                1 if data.get("log_level", "").lower() in ["error", "fatal"] else 0
            )
            project_score = 0.052209
            handled = data.get("handled", 0)

            input_data = np.append(
                embeddings.reshape(1, -1),
                [[has_stacktrace, log_level_error, project_score, handled]],
                axis=1,
            )

            pred = self.classifier.predict(input_data)[0]
        return min(1.0, max(0.0, pred))
