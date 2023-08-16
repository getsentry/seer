import pandas as pd
import numpy as np
import torch
from transformers import BertForSequenceClassification, AdamW, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from joblib import dump


class SeverityTraining:
    def __init__(self, config=None):
        """Initialize the training class with configuration."""
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.config = config or {
            "batch_size": 16,
            "max_len": 128,
            "epochs": 3,
            "lr": 2e-5,
            "score_col": "score",
        }

    class ImportanceDataset(Dataset):
        def __init__(self, messages, scores, tokenizer, max_len):
            self.messages = messages
            self.scores = scores
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.messages)

        def __getitem__(self, item):
            message = str(self.messages[item])
            score = self.scores[item]
            encoding = self.tokenizer.encode_plus(
                message,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "message_text": message,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "scores": torch.tensor(
                    score, dtype=torch.long
                ),  # Change to long for classification
            }

    def read_data(self):
        # TODO: need to replace this with something that interfaces with GCS
        bq_data = pd.read_csv("datasets/sentry/bq_data.csv")
        bq_scores = pd.read_csv("datasets/sentry/bq_scores.csv").rename(
            columns={"id": "group_id"}
        )
        data = bq_data.merge(bq_scores, on="group_id").groupby("group_id").first()
        return data

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = self.ImportanceDataset(
            messages=df.message.to_numpy(),
            scores=df[self.config["score_col"]].to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=0)

    def train_epoch(self, model, data_loader, optimizer):
        model = model.train()
        losses = []
        for d in data_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["scores"].to(self.device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=targets
            )
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def eval_embedding_model(self, model, data_loader):
        model = model.eval()
        losses = []
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["scores"].to(self.device)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=targets
                )
                loss = outputs[0]
                losses.append(loss.item())
        return sum(losses) / len(losses)

    def tune_embeddings_model(self, X_train, y_train):
        """Tune the embeddings model with training data."""
        X_train["label"] = y_train
        train_df, val_df = train_test_split(X_train, test_size=0.2, random_state=42)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        train_data_loader = self.create_data_loader(
            train_df, tokenizer, self.config["max_len"], self.config["batch_size"]
        )
        val_data_loader = self.create_data_loader(
            val_df, tokenizer, self.config["max_len"], self.config["batch_size"]
        )
        optimizer = AdamW(model.parameters(), lr=self.config["lr"])
        model = model.to(self.device)
        for epoch in range(self.config["epochs"]):
            print(f'Epoch {epoch + 1}/{self.config["epochs"]}')
            print("-" * 10)
            train_loss = self.train_epoch(model, train_data_loader, optimizer)
            print(f"Train loss {train_loss}")
            val_loss = self.eval_embedding_model(model, val_data_loader)
            print(f"Val loss {val_loss}")
        return model, tokenizer

    def generate_feature_df(
        self,
        df,
        model,
        tokenizer,
        extra_features=["has_stacktrace", "event_size"],
        use_pca=False,
    ):
        """Generate a feature DataFrame for logistic regression."""
        embeddings = []
        for val in df["message"]:
            embeddings.append(self.get_embeddings(model, tokenizer, val))
        if len(extra_features) > 0:
            model_df = pd.concat(
                [
                    pd.DataFrame(embeddings),
                    df[extra_features]
                    .replace({True: 1, False: 0})
                    .reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            model_df = pd.DataFrame(embeddings)
        if use_pca:
            pca = PCA(n_components=200)
            model_df = pd.DataFrame(pca.fit_transform(model_df))
        model_df.columns = model_df.columns.astype(str)
        return model_df

    def get_embeddings(self, model, tokenizer, text, max_len=128):
        """Generate embeddings for the given text using the pre-trained model."""
        model = model.to(self.device)
        inputs = tokenizer.encode_plus(
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

    def train_classification_model(self, X_train, y_train, save_path=None):
        """Train the classification model and optionally save it."""
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        if save_path:
            dump(lr, save_path)
        return lr
