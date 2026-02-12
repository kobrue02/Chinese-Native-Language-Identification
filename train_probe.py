"""Extract frozen embeddings from a transformer and train a linear probe.

No transformer layers are fine-tuned — the model is used purely as a
feature extractor, and a LogisticRegression is trained on the output.

Usage:
    python train_probe.py --model google-bert/bert-base-chinese
    python train_probe.py --model Qwen/Qwen3-Embedding-4B --batch-size 4
"""

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

import config
from data_loader import load_and_split
from evaluate import evaluate_and_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tokenisation dataset ──────────────────────────────────────────────────

class TextDataset(Dataset):
    """Wraps pre-tokenised encodings (no labels needed)."""

    def __init__(self, encodings: dict):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def tokenize_texts(texts: list[str], tokenizer, max_length: int, desc: str) -> dict:
    """Tokenize in batches of 256 with a progress bar."""
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
    result = {
        "input_ids": torch.cat(all_input_ids),
        "attention_mask": torch.cat(all_attention_mask),
    }
    if all_token_type_ids:
        result["token_type_ids"] = torch.cat(all_token_type_ids)
    return result


# ── Embedding extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: AutoModel,
    encodings: dict,
    batch_size: int,
    desc: str = "Extracting embeddings",
) -> np.ndarray:
    """Run the frozen encoder and mean-pool into fixed-size vectors."""
    model.eval()
    dataset = TextDataset(encodings)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_embeds = []

    for batch in trange(len(loader), desc=desc):
        # manual iteration to use trange
        pass

    # re-iterate properly with progress
    all_embeds = []
    loader_iter = iter(loader)
    for _ in trange(len(loader), desc=desc):
        batch = next(loader_iter)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        attention_mask = batch["attention_mask"]
        outputs = model(**batch)
        hidden = outputs.last_hidden_state  # (B, seq, hidden)
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embeds.append(pooled.cpu().numpy())

    return np.concatenate(all_embeds)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=config.ENCODER_MODELS[0]["name"],
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (lower for large models)",
    )
    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size

    print(f"Device: {DEVICE}")
    print(f"Model:  {model_name} (frozen — linear probe only)")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_df, val_df, test_df, le = load_and_split()
    label_names = list(le.classes_)

    # ── Load model & tokenizer ────────────────────────────────────────────
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)

    # ── Tokenize ──────────────────────────────────────────────────────────
    max_length = config.TRANSFORMER_CONFIG["max_length"]
    train_enc = tokenize_texts(train_df["text"].tolist(), tokenizer, max_length, "Tokenizing train")
    val_enc = tokenize_texts(val_df["text"].tolist(), tokenizer, max_length, "Tokenizing val")
    test_enc = tokenize_texts(test_df["text"].tolist(), tokenizer, max_length, "Tokenizing test")

    # ── Extract embeddings ────────────────────────────────────────────────
    X_train = extract_embeddings(model, train_enc, batch_size, "Encoding train")
    X_val = extract_embeddings(model, val_enc, batch_size, "Encoding val")
    X_test = extract_embeddings(model, test_enc, batch_size, "Encoding test")
    print(f"Embedding dim: {X_train.shape[1]}")

    # Free GPU memory — we only need numpy arrays from here
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # ── Train linear probe ────────────────────────────────────────────────
    print("Training LogisticRegression probe...")
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        random_state=config.RANDOM_SEED,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────
    safe_name = "probe_" + model_name.replace("/", "_")

    y_val_pred = clf.predict(X_val)
    print("\n-- Validation --")
    evaluate_and_report(y_val, y_val_pred, label_names, safe_name + "_val")

    y_test_pred = clf.predict(X_test)
    print("\n-- Test --")
    evaluate_and_report(y_test, y_test_pred, label_names, safe_name + "_test")


if __name__ == "__main__":
    main()
