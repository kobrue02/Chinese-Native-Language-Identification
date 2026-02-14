"""Run frozen-embedding probes for ALL encoder models sequentially.

Loads each model one at a time, extracts embeddings, trains a
LogisticRegression, evaluates, then frees memory before the next model.
Models that fail to load are skipped gracefully.

Usage:
    python train_probe_all.py
    python train_probe_all.py --batch-size 16
"""

import argparse
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer

import config
from data_loader import load_and_split
from evaluate import compute_metrics
from train_probe import extract_embeddings, tokenize_texts


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_probe(
    model_name: str,
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    batch_size: int,
) -> dict:
    """Run a single probe and return metrics dict. Raises on failure."""
    max_length = config.TRANSFORMER_CONFIG["max_length"]

    # Load
    print(f"  Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)

    # Tokenize
    train_enc = tokenize_texts(train_texts, tokenizer, max_length, "  Tokenizing train")
    val_enc = tokenize_texts(val_texts, tokenizer, max_length, "  Tokenizing val")
    test_enc = tokenize_texts(test_texts, tokenizer, max_length, "  Tokenizing test")

    # Extract embeddings
    X_train = extract_embeddings(model, train_enc, batch_size, "  Encoding train")
    X_val = extract_embeddings(model, val_enc, batch_size, "  Encoding val")
    X_test = extract_embeddings(model, test_enc, batch_size, "  Encoding test")
    print(f"  Embedding dim: {X_train.shape[1]}")

    # Free GPU memory
    del model, tokenizer, train_enc, val_enc, test_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train probe
    print("  Training MLP probe...")
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=config.RANDOM_SEED,
        verbose=False,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    val_metrics = compute_metrics(y_val, clf.predict(X_val), label_names)
    test_metrics = compute_metrics(y_test, clf.predict(X_test), label_names)

    return {
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "embedding_dim": X_train.shape[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # Load data once
    print("Loading data...")
    train_df, val_df, test_df, le = load_and_split()
    label_names = list(le.classes_)
    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Run each model
    rows = []
    for i, entry in enumerate(config.ENCODER_MODELS):
        model_name = entry["name"]
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(config.ENCODER_MODELS)}] {model_name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            metrics = run_probe(
                model_name,
                train_texts, val_texts, test_texts,
                y_train, y_val, y_test,
                label_names,
                args.batch_size,
            )
            elapsed = time.time() - t0
            rows.append({"model": model_name, **metrics, "time_s": f"{elapsed:.0f}"})
            print(f"  Done in {elapsed:.0f}s — "
                  f"test acc={metrics['test_accuracy']:.4f}, "
                  f"test F1={metrics['test_macro_f1']:.4f}")
        except Exception:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.0f}s:")
            traceback.print_exc()
            rows.append({
                "model": model_name,
                "val_accuracy": None, "val_macro_f1": None, "val_weighted_f1": None,
                "test_accuracy": None, "test_macro_f1": None, "test_weighted_f1": None,
                "embedding_dim": None, "time_s": "FAIL",
            })

    # Summary table
    results = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print("SUMMARY — Linear Probe Results")
    print(f"{'=' * 60}")

    display_cols = ["model", "embedding_dim", "test_accuracy", "test_macro_f1", "test_weighted_f1", "time_s"]
    print(results[display_cols].to_string(index=False, float_format="%.4f"))

    # Save CSV
    csv_path = config.RESULTS_DIR / "probe_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Save LaTeX
    valid = results.dropna(subset=["test_accuracy"])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MLP probe results (frozen embeddings + MLPClassifier).}",
        r"\label{tab:probe-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Dim} & \textbf{Val Acc} & \textbf{Val F1} & \textbf{Test Acc} & \textbf{Test F1} & \textbf{Time (s)} \\",
        r"\midrule",
    ]
    for _, r in valid.iterrows():
        name = r["model"].replace("_", r"\_")
        lines.append(
            f"{name} & {int(r['embedding_dim'])} & {r['val_accuracy']:.4f} & "
            f"{r['val_macro_f1']:.4f} & {r['test_accuracy']:.4f} & "
            f"{r['test_macro_f1']:.4f} & {r['time_s']} \\\\"
        )
    if len(valid) < len(results):
        lines.append(r"\midrule")
        for _, r in results[results["test_accuracy"].isna()].iterrows():
            name = r["model"].replace("_", r"\_")
            lines.append(f"{name} & \\multicolumn{{6}}{{c}}{{failed to load}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    tex_path = config.RESULTS_DIR / "probe_results.tex"
    tex_path.write_text("\n".join(lines))
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
