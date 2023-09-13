import sentry_sdk
import torch
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
        model = self.embeddings_model.to(self.device)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = model.bert(inputs["input_ids"], inputs["attention_mask"])
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze()

        embeddings = embeddings.cpu().numpy()
        return embeddings

    def severity_score(self, text):
        """Predict the severity score for the given text using the pre-trained classifier."""
        with sentry_sdk.start_span(
            op="model.severity", description="get_embeddings"
        ):
            embeddings = self.get_embeddings(text)
        with sentry_sdk.start_span(
            op="model.severity", description="predict_proba"
        ):
            pred = self.classifier.predict(embeddings.reshape(1, -1))[0]
        return min(1.0, max(0.0, pred))
