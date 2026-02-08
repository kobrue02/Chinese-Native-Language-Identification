import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from data_loader import load_corpus


def plot_language_distribution(df: pd.DataFrame) -> None:
    """Bar plot of samples per native language (sorted, with log-scale option)."""
    counts = df["native_language"].value_counts().sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(counts) * 0.3)))

    # Linear scale
    counts.plot.barh(ax=axes[0])
    axes[0].set_title("Samples per Native Language")
    axes[0].set_xlabel("Count")

    # Log scale
    counts.plot.barh(ax=axes[1])
    axes[1].set_xscale("log")
    axes[1].set_title("Samples per Native Language (log scale)")
    axes[1].set_xlabel("Count (log)")

    plt.tight_layout()
    fig.savefig(config.RESULTS_DIR / "language_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved language_distribution.png")


def plot_context_distribution(df: pd.DataFrame) -> None:
    """Bar plot of samples per writing context."""
    counts = df["context"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax)
    ax.set_title("Samples per Writing Context")
    ax.set_ylabel("Count")
    ax.set_xlabel("Context")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(config.RESULTS_DIR / "context_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved context_distribution.png")


def plot_text_lengths(df: pd.DataFrame) -> None:
    """Histogram of text lengths in characters and words (via jieba)."""
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].apply(lambda t: len(list(jieba.cut(t))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["char_len"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title("Text Length (characters)")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df["word_len"], bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_title("Text Length (words, jieba)")
    axes[1].set_xlabel("Words")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    fig.savefig(config.RESULTS_DIR / "text_lengths.png", dpi=150)
    plt.close(fig)
    print("Saved text_lengths.png")


def plot_language_context_heatmap(df: pd.DataFrame) -> None:
    """Cross-tab heatmap of native language × writing context."""
    ct = pd.crosstab(df["native_language"], df["context"])
    # Sort by total samples
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(ct) * 0.35)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Native Language × Writing Context")
    plt.tight_layout()
    fig.savefig(config.RESULTS_DIR / "language_context_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved language_context_heatmap.png")


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print basic corpus statistics."""
    print(f"\nTotal documents: {len(df)}")
    print(f"Unique native languages: {df['native_language'].nunique()}")
    print(f"Writing contexts: {list(df['context'].unique())}")
    print(
        f"\nLanguage distribution:\n{df['native_language'].value_counts().to_string()}"
    )
    print(f"\nGender distribution:\n{df['gender'].value_counts().to_string()}")


if __name__ == "__main__":
    print("Loading corpus...")
    df = load_corpus()
    print_summary_stats(df)
    print("\nGenerating plots...")
    plot_language_distribution(df)
    plot_context_distribution(df)
    plot_text_lengths(df)
    plot_language_context_heatmap(df)
    print("\nAll plots saved to", config.RESULTS_DIR)
