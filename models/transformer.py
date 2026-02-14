import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

CFG = config.TRANSFORMER_CONFIG


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
    return model, tokenizer


class NLIDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for transformer-based NLI."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
