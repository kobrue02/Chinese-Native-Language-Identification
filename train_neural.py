"""Train and evaluate LSTM/CNN models for NLI."""

import time
from collections import Counter

import jieba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
<<<<<<< HEAD
=======
from tqdm import tqdm
>>>>>>> e689c55 (add more scripts and features)

import config
from data_loader import load_and_split
from evaluate import evaluate_and_report
from models.neural import BiLSTMClassifier, TextCNN

CFG = config.NEURAL_CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Vocabulary ────────────────────────────────────────────────────────────


def build_vocab(texts: list[str], max_size: int) -> dict[str, int]:
    """Build a word-to-index vocabulary from training texts using jieba."""
    counter: Counter[str] = Counter()
<<<<<<< HEAD
    for text in texts:
=======
    for text in tqdm(texts, desc="Building vocab"):
>>>>>>> e689c55 (add more scripts and features)
        counter.update(jieba.cut(text))
    # 0 = pad, 1 = unk
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_indices(
    texts: list[str], vocab: dict[str, int], max_len: int
) -> np.ndarray:
    """Convert texts to padded/truncated index sequences."""
    result = np.zeros((len(texts), max_len), dtype=np.int64)
    unk_idx = vocab["<UNK>"]
<<<<<<< HEAD
    for i, text in enumerate(texts):
=======
    for i, text in enumerate(tqdm(texts, desc="Tokenizing")):
>>>>>>> e689c55 (add more scripts and features)
        tokens = list(jieba.cut(text))[:max_len]
        for j, tok in enumerate(tokens):
            result[i, j] = vocab.get(tok, unk_idx)
    return result


# ── Dataset ───────────────────────────────────────────────────────────────


class NLIDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.array(X))
        self.y = torch.from_numpy(np.array(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training loop ─────────────────────────────────────────────────────────


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


<<<<<<< HEAD
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
=======
def train_epoch(model, loader, criterion, optimizer, epoch=0):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"  Train epoch {epoch}", leave=False)
    for X_batch, y_batch in pbar:
>>>>>>> e689c55 (add more scripts and features)
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
<<<<<<< HEAD
=======
        pbar.set_postfix(loss=f"{loss.item():.4f}")
>>>>>>> e689c55 (add more scripts and features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
<<<<<<< HEAD
def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
=======
def evaluate_epoch(model, loader, criterion, desc="Eval"):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in tqdm(loader, desc=f"  {desc}", leave=False):
>>>>>>> e689c55 (add more scripts and features)
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def train_model(model, train_loader, val_loader, criterion):
    """Train with early stopping on validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
<<<<<<< HEAD
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, criterion)
=======
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_preds, val_labels = evaluate_epoch(model, val_loader, criterion, "Val")
>>>>>>> e689c55 (add more scripts and features)
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
    return model


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    print(f"Device: {DEVICE}")

    # Load data
    print("Loading data...")
    train_df, val_df, test_df, le = load_and_split()
    label_names = list(le.classes_)
    num_classes = len(label_names)

    # Build vocabulary from training set
    print("Building vocabulary...")
    vocab = build_vocab(train_df["text"].tolist(), CFG["vocab_size"])
    print(f"Vocabulary size: {len(vocab)}")

    # Convert to index sequences
    print("Tokenizing...")
    X_train = texts_to_indices(train_df["text"].tolist(), vocab, CFG["max_seq_len"])
    X_val = texts_to_indices(val_df["text"].tolist(), vocab, CFG["max_seq_len"])
    X_test = texts_to_indices(test_df["text"].tolist(), vocab, CFG["max_seq_len"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # DataLoaders
    train_loader = DataLoader(
        NLIDataset(X_train, y_train), batch_size=CFG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(NLIDataset(X_val, y_val), batch_size=CFG["batch_size"])
    test_loader = DataLoader(NLIDataset(X_test, y_test), batch_size=CFG["batch_size"])

    # Class weights for imbalanced data
    class_weights = compute_class_weights(y_train, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Train BiLSTM ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training BiLSTM")
    print("=" * 60)
    lstm_model = BiLSTMClassifier(len(vocab), num_classes).to(DEVICE)
    lstm_model = train_model(lstm_model, train_loader, val_loader, criterion)

<<<<<<< HEAD
    _, test_preds, test_labels = evaluate_epoch(lstm_model, test_loader, criterion)
=======
    _, test_preds, test_labels = evaluate_epoch(lstm_model, test_loader, criterion, "Test")
>>>>>>> e689c55 (add more scripts and features)
    evaluate_and_report(test_labels, test_preds, label_names, "BiLSTM")

    # ── Train TextCNN ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training TextCNN")
    print("=" * 60)
    cnn_model = TextCNN(len(vocab), num_classes).to(DEVICE)
    cnn_model = train_model(cnn_model, train_loader, val_loader, criterion)

<<<<<<< HEAD
    _, test_preds, test_labels = evaluate_epoch(cnn_model, test_loader, criterion)
=======
    _, test_preds, test_labels = evaluate_epoch(cnn_model, test_loader, criterion, "Test")
>>>>>>> e689c55 (add more scripts and features)
    evaluate_and_report(test_labels, test_preds, label_names, "TextCNN")


if __name__ == "__main__":
    main()
