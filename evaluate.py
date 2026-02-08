import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Compute accuracy, macro-F1, weighted-F1, and per-class report."""
    # Only report on labels actually present in y_true or y_pred
    present = sorted(set(y_true) | set(y_pred))
    names = [label_names[i] for i in present] if label_names else None

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=present,
        target_names=names,
        zero_division=0,
        output_dict=True,
    )
    metrics["classification_report"] = report
    return metrics


def print_metrics(metrics: dict, model_name: str = "") -> None:
    """Print key metrics to stdout."""
    header = f"Results: {model_name}" if model_name else "Results"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:  {metrics['weighted_f1']:.4f}")


def save_report(metrics: dict, path) -> None:
    """Save full classification report to a JSON file."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    model_name: str = "",
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix."""
    present = sorted(set(y_true) | set(y_pred))
    label_names = [label_names[i] for i in present]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(
        figsize=(max(10, len(label_names) * 0.5), max(8, len(label_names) * 0.4))
    )
    sns.heatmap(
        cm,
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
        fmt=".2f" if normalize else "d",
        annot=len(label_names) <= 25,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = "Confusion Matrix"
    if model_name:
        title += f" — {model_name}"
    if normalize:
        title += " (normalized)"
    ax.set_title(title)
    plt.tight_layout()

    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(config.RESULTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def evaluate_and_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    model_name: str,
) -> dict:
    """Full evaluation pipeline: metrics, print, save, plot."""
    metrics = compute_metrics(y_true, y_pred, label_names)
    print_metrics(metrics, model_name)
    save_report(
        metrics,
        config.RESULTS_DIR / f"report_{model_name.lower().replace(' ', '_')}.json",
    )
    plot_confusion_matrix(y_true, y_pred, label_names, model_name)
    return metrics
