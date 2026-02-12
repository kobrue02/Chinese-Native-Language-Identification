"""Fine-tune and evaluate a transformer model for NLI."""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import config
from data_loader import load_and_split
from evaluate import evaluate_and_report
from models.transformer import NLIDataset, load_model_and_tokenizer

CFG = config.TRANSFORMER_CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch)
        total_loss += criterion(outputs.logits, labels).item() * len(labels)
        all_preds.append(outputs.logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def main():
    print(f"Device: {DEVICE}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_df, val_df, test_df, le = load_and_split()
    label_names = list(le.classes_)
    num_classes = len(label_names)

    # ── Load model & tokenizer ────────────────────────────────────────────
    model_name = CFG["model_name"]
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(num_classes, model_name)
    model = model.to(DEVICE)

    # ── Create datasets ───────────────────────────────────────────────────
    print("Tokenizing texts...")
    train_dataset = NLIDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        CFG["max_length"],
    )
    val_dataset = NLIDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        CFG["max_length"],
    )
    test_dataset = NLIDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        CFG["max_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CFG["batch_size"])

    # ── Loss with class weights ───────────────────────────────────────────
    class_weights = compute_class_weights(train_df["label"].values, num_classes).to(
        DEVICE
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion)
        elapsed = time.time() - t0

        from sklearn.metrics import f1_score

        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  ({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    # ── Evaluate on test set ──────────────────────────────────────────────
    _, test_preds, test_labels = evaluate_model(model, test_loader, criterion)
    safe_name = model_name.replace("/", "_")
    evaluate_and_report(test_labels, test_preds, label_names, safe_name)


if __name__ == "__main__":
    main()
