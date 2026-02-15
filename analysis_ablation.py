"""Feature-group ablation study for MLP on hand-crafted features.

Extracts all 9 feature groups once, then runs 19 classifier trainings:
  - 1 full baseline (all 9 groups)
  - 9 leave-one-out (drop one group at a time)
  - 9 individual (each group alone)

Pipeline per run: hstack → TruncatedSVD → StandardScaler → MLPClassifier

Usage:
    python analysis_ablation.py
"""

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_and_split
from evaluate import compute_metrics
from features.ngrams import char_ngram_vectorizer, word_ngram_vectorizer
from features.pos_tags import extract_pos_features
from features.pos_ngrams import pos_ngram_vectorizer, pos_to_sequences
from features.function_words import extract_function_word_features
from features.particles import extract_particle_features
from features.discourse import extract_discourse_features
from features.lexical_richness import extract_lexical_richness_features
from features.segmentation import extract_segmentation_features

RESULTS = config.RESULTS_DIR

# Feature group metadata: (key, display name)
GROUPS = OrderedDict([
    ("char",   "Char n-grams"),
    ("word",   "Word n-grams"),
    ("pos",    "POS tags"),
    ("pos_ng", "POS n-grams"),
    ("func",   "Function words"),
    ("part",   "Particles"),
    ("disc",   "Discourse"),
    ("lex",    "Lexical richness"),
    ("seg",    "Segmentation"),
])


# ── Step 1: Extract all 9 feature groups ──────────────────────────────────

def extract_all_groups(train_texts, test_texts):
    """Extract each feature group independently.

    Returns dict of {key: (X_train_sparse, X_test_sparse)}.
    """
    blocks = {}

    def _sparse(arr):
        return sp.csr_matrix(arr)

    # 1. Character n-grams
    print("  [1/9] Character n-gram TF-IDF...")
    vec = char_ngram_vectorizer()
    blocks["char"] = (vec.fit_transform(train_texts), vec.transform(test_texts))

    # 2. Word n-grams (jieba)
    print("  [2/9] Word n-gram TF-IDF (jieba)...")
    word_vec, jieba_tok = word_ngram_vectorizer()
    jieba_tok.set_total(len(train_texts), desc="    fit (train)")
    tr = word_vec.fit_transform(train_texts)
    jieba_tok.close()
    jieba_tok.set_total(len(test_texts), desc="    transform (test)")
    te = word_vec.transform(test_texts)
    jieba_tok.close()
    blocks["word"] = (tr, te)

    # 3. POS tag distribution
    print("  [3/9] POS tag distributions...")
    blocks["pos"] = (
        _sparse(extract_pos_features(train_texts)),
        _sparse(extract_pos_features(test_texts)),
    )

    # 4. POS n-grams
    print("  [4/9] POS n-gram TF-IDF...")
    train_seq = pos_to_sequences(train_texts, desc="    POS seq (train)")
    test_seq = pos_to_sequences(test_texts, desc="    POS seq (test)")
    pv = pos_ngram_vectorizer()
    blocks["pos_ng"] = (pv.fit_transform(train_seq), pv.transform(test_seq))

    # 5. Function words
    print("  [5/9] Function word frequencies...")
    blocks["func"] = (
        _sparse(extract_function_word_features(train_texts)),
        _sparse(extract_function_word_features(test_texts)),
    )

    # 6. Particles
    print("  [6/9] Particle context features...")
    blocks["part"] = (
        _sparse(extract_particle_features(train_texts)),
        _sparse(extract_particle_features(test_texts)),
    )

    # 7. Discourse
    print("  [7/9] Discourse connectives & sentence features...")
    blocks["disc"] = (
        _sparse(extract_discourse_features(train_texts)),
        _sparse(extract_discourse_features(test_texts)),
    )

    # 8. Lexical richness
    print("  [8/9] Lexical richness features...")
    blocks["lex"] = (
        _sparse(extract_lexical_richness_features(train_texts)),
        _sparse(extract_lexical_richness_features(test_texts)),
    )

    # 9. Segmentation
    print("  [9/9] Segmentation-derived features...")
    blocks["seg"] = (
        _sparse(extract_segmentation_features(train_texts)),
        _sparse(extract_segmentation_features(test_texts)),
    )

    # Report dimensions
    for key, name in GROUPS.items():
        dims = blocks[key][0].shape[1]
        print(f"    {name:<20s} {dims:>6d} dims")

    return blocks


# ── Step 2: Combine and classify ──────────────────────────────────────────

def combine_blocks(blocks, keys):
    """hstack selected feature blocks into a single sparse matrix pair."""
    train_parts = [blocks[k][0] for k in keys]
    test_parts = [blocks[k][1] for k in keys]
    return (
        sp.hstack(train_parts, format="csr"),
        sp.hstack(test_parts, format="csr"),
    )


def train_and_predict(X_train, y_train, X_test, run_label=""):
    """TruncatedSVD → StandardScaler → MLPClassifier. Returns y_pred."""
    n_components = min(300, X_train.shape[1], X_train.shape[0])
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    X_tr_svd = svd.fit_transform(X_train)
    X_te_svd = svd.transform(X_test)
    explained = svd.explained_variance_ratio_.sum()

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_svd)
    X_te_sc = scaler.transform(X_te_svd)

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        early_stopping=True,
        random_state=config.RANDOM_SEED,
        max_iter=300,
    )
    clf.fit(X_tr_sc, y_train)

    label = f"  {run_label}: " if run_label else "  "
    print(f"{label}{X_train.shape[1]} feats → {n_components} SVD dims "
          f"({explained:.1%} var), MLP stopped at epoch {clf.n_iter_}")

    return clf.predict(X_te_sc)


# ── Step 3: Run ablation experiments ──────────────────────────────────────

def run_ablations(blocks, y_train, y_test, label_names):
    """Run full, leave-one-out, and individual experiments."""
    keys = list(GROUPS.keys())
    results = {}

    # --- Full baseline ---
    print("\n── Full baseline (all 9 groups) ──")
    X_tr, X_te = combine_blocks(blocks, keys)
    y_pred = train_and_predict(X_tr, y_train, X_te, "ALL")
    m = compute_metrics(y_test, y_pred, label_names)
    results["full"] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    print(f"  → Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")

    full_f1 = m["macro_f1"]

    # --- Leave-one-out ---
    print("\n── Leave-one-out (drop one group) ──")
    loo_results = {}
    for drop_key in keys:
        subset = [k for k in keys if k != drop_key]
        X_tr, X_te = combine_blocks(blocks, subset)
        y_pred = train_and_predict(X_tr, y_train, X_te, f"LOO-{drop_key}")
        m = compute_metrics(y_test, y_pred, label_names)
        delta = m["macro_f1"] - full_f1
        loo_results[drop_key] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "delta_f1": delta,
        }
        print(f"  → Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  ΔF1={delta:+.4f}")
    results["loo"] = loo_results

    # --- Individual groups ---
    print("\n── Individual groups (each alone) ──")
    ind_results = {}
    for key in keys:
        X_tr, X_te = combine_blocks(blocks, [key])
        y_pred = train_and_predict(X_tr, y_train, X_te, f"SOLO-{key}")
        m = compute_metrics(y_test, y_pred, label_names)
        ind_results[key] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
        }
        print(f"  → Acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}")
    results["individual"] = ind_results

    return results


# ── Step 4: Output tables and charts ──────────────────────────────────────

def print_table(results):
    """Print a formatted console table."""
    full_f1 = results["full"]["macro_f1"]
    full_acc = results["full"]["accuracy"]

    print("\n" + "=" * 90)
    print("Feature Ablation Results")
    print("=" * 90)
    print(f"  Full baseline: Acc={full_acc:.4f}  F1={full_f1:.4f}")
    print()
    print(f"  {'Group':<20s} {'Ind Acc':>8s} {'Ind F1':>8s} "
          f"{'LOO Acc':>8s} {'LOO F1':>8s} {'ΔF1':>8s}")
    print("  " + "-" * 68)

    for key, name in GROUPS.items():
        ind = results["individual"][key]
        loo = results["loo"][key]
        print(f"  {name:<20s} {ind['accuracy']:>8.4f} {ind['macro_f1']:>8.4f} "
              f"{loo['accuracy']:>8.4f} {loo['macro_f1']:>8.4f} "
              f"{loo['delta_f1']:>+8.4f}")


def save_bar_chart(results):
    """Horizontal bar chart: LOO F1 drop per group, sorted by impact."""
    loo = results["loo"]
    # Sort by delta (most negative = biggest impact first)
    sorted_keys = sorted(loo.keys(), key=lambda k: loo[k]["delta_f1"])
    names = [GROUPS[k] for k in sorted_keys]
    deltas = [loo[k]["delta_f1"] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas]
    y_pos = range(len(names))
    ax.barh(y_pos, deltas, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ΔF1 (drop from full model)")
    ax.set_title("Feature Group Importance (Leave-One-Out F1 Drop)")
    ax.axvline(0, color="black", linewidth=0.5)

    # Annotate values
    for i, d in enumerate(deltas):
        ha = "right" if d < 0 else "left"
        offset = -0.001 if d < 0 else 0.001
        ax.text(d + offset, i, f"{d:+.4f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    path_png = RESULTS / "ablation_feature_importance.png"
    path_pdf = RESULTS / "ablation_feature_importance.pdf"
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path_png}")


def save_latex_table(results):
    """Save a LaTeX table of ablation results."""
    full_f1 = results["full"]["macro_f1"]
    full_acc = results["full"]["accuracy"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Feature group ablation study (MLP classifier). "
        r"Full model: Acc=%.4f, F1=%.4f.}" % (full_acc, full_f1),
        r"\label{tab:feature-ablation}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Feature Group} & \textbf{Ind.\ Acc} & \textbf{Ind.\ F1} "
        r"& \textbf{LOO Acc} & \textbf{LOO F1} & \textbf{$\Delta$F1} \\",
        r"\midrule",
    ]
    for key, name in GROUPS.items():
        ind = results["individual"][key]
        loo = results["loo"][key]
        lines.append(
            f"{name} & {ind['accuracy']:.4f} & {ind['macro_f1']:.4f} "
            f"& {loo['accuracy']:.4f} & {loo['macro_f1']:.4f} "
            f"& {loo['delta_f1']:+.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = RESULTS / "ablation_feature_importance.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved {tex_path}")


def save_json(results):
    """Save raw ablation results as JSON."""
    path = RESULTS / "ablation_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Feature Ablation Study — MLP on Hand-Crafted Features")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df, _, test_df, le = load_and_split()
    label_names = list(le.classes_)
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}, "
          f"Classes: {len(label_names)}")

    # Step 1: Extract all feature groups
    print("\nExtracting all 9 feature groups...")
    blocks = extract_all_groups(train_texts, test_texts)

    # Step 2-3: Run ablation experiments
    print("\nRunning 19 ablation experiments (1 full + 9 LOO + 9 individual)...")
    results = run_ablations(blocks, y_train, y_test, label_names)

    # Step 4: Output
    print_table(results)
    print("\nSaving outputs...")
    save_bar_chart(results)
    save_latex_table(results)
    save_json(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
