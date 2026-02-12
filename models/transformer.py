<<<<<<< HEAD
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
=======
"""Transformer model loading for both classifier and embedding models."""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
>>>>>>> e689c55 (add more scripts and features)

import config

CFG = config.TRANSFORMER_CONFIG


<<<<<<< HEAD
def load_model_and_tokenizer(
    num_labels: int,
    model_name: str | None = None,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a pre-trained transformer and its tokenizer."""
    model_name = model_name or CFG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
=======
def get_model_config(model_name: str) -> dict:
    """Look up model config from ENCODER_MODELS list."""
    for m in config.ENCODER_MODELS:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Unknown model: {model_name}. Add it to config.ENCODER_MODELS.")


class EmbeddingClassifier(nn.Module):
    """Wraps an embedding model with a classification head.

    Uses mean pooling over token embeddings, then a linear layer.
    """

    def __init__(self, encoder: AutoModel, hidden_size: int, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, **kwargs):
        labels = kwargs.pop("labels", None)
        outputs = self.encoder(**kwargs)
        pooled = self._mean_pool(outputs.last_hidden_state, kwargs["attention_mask"])
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return type("Output", (), {"logits": logits, "loss": loss})()


def load_model_and_tokenizer(
    num_labels: int,
    model_name: str,
) -> tuple[nn.Module, AutoTokenizer]:
    """Load a model and tokenizer. Handles both classifier and embedding types."""
    mcfg = get_model_config(model_name)
    model_type = mcfg["type"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model_type == "classifier":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True,
        )
    else:
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = encoder.config.hidden_size
        model = EmbeddingClassifier(encoder, hidden_size, num_labels)

>>>>>>> e689c55 (add more scripts and features)
    return model, tokenizer


class NLIDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for transformer-based NLI."""

<<<<<<< HEAD
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
=======
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
        desc: str = "Tokenizing",
    ):
        from tqdm import trange

        # Tokenize in batches of 256 with a progress bar
        batch_size = 256
        all_input_ids, all_attention_mask, all_token_type_ids = [], [], []
        for i in trange(0, len(texts), batch_size, desc=desc):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            all_input_ids.append(enc["input_ids"])
            all_attention_mask.append(enc["attention_mask"])
            if "token_type_ids" in enc:
                all_token_type_ids.append(enc["token_type_ids"])
        self.encodings = {
            "input_ids": torch.cat(all_input_ids),
            "attention_mask": torch.cat(all_attention_mask),
        }
        if all_token_type_ids:
            self.encodings["token_type_ids"] = torch.cat(all_token_type_ids)
>>>>>>> e689c55 (add more scripts and features)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
