"""Deeper analyses for the discussion section.

Produces charts and tables answering four questions:
  Q1. Which L1s benefit most from LERT?
  Q2. Does POS-heavy pretraining correlate with syntactic-transfer signals?
  Q3. Are radical features implicitly captured by encoder models?
  Q4. Is whole-word masking helpful for NLI?

Usage:
    python analysis_discussion.py              # Q1 + Q4 (fast, JSON only)
    python analysis_discussion.py --all        # all four (trains SVMs)
    python analysis_discussion.py --pos-only   # Q2 only
    python analysis_discussion.py --radicals   # Q3 only
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

RESULTS = config.RESULTS_DIR
MIN_SUPPORT = 10  # ignore L1s with fewer test samples

# ── Typological grouping ────────────────────────────────────────────────
# SOV languages whose speakers are expected to show word-order transfer
SOV_LANGUAGES = {"South Korea", "Japan", "Mongolia", "Turkey"}
# SVO languages (like Chinese) — less syntactic transfer expected
SVO_LANGUAGES = {"Vietnam", "Thailand", "Cambodia", "Laos",
                 "Indonesia", "Philippines", "Myanmar"}


# ── Helpers ─────────────────────────────────────────────────────────────

def load_report(name: str) -> dict | None:
    path = RESULTS / f"report_{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


_SKIP_KEYS = {"accuracy", "macro avg", "weighted avg"}


def per_class_f1(report: dict) -> dict[str, float]:
    """Extract {L1: F1} from a report, filtering by support."""
    cr = report["classification_report"]
    out = {}
    for lang, vals in cr.items():
        if lang in _SKIP_KEYS:
            continue
        if not isinstance(vals, dict) or "f1-score" not in vals:
            continue
        if vals.get("support", 0) >= MIN_SUPPORT:
            out[lang] = vals["f1-score"]
    return out


def per_class_f1_all_support(report: dict) -> dict[str, tuple[float, int]]:
    """Extract {L1: (F1, support)} from a report."""
    cr = report["classification_report"]
    out = {}
    for lang, vals in cr.items():
        if lang in _SKIP_KEYS:
            continue
        if not isinstance(vals, dict) or "f1-score" not in vals:
            continue
        sup = int(vals.get("support", 0))
        if sup >= MIN_SUPPORT:
            out[lang] = (vals["f1-score"], sup)
    return out


SVD_COMPONENTS = 300  # default dimensionality after reduction


def reduce_and_train(X_train, y_train, X_test, n_components=None):
    """TruncatedSVD → StandardScaler → SGDClassifier.  Fast on sparse data."""
    from sklearn.linear_model import SGDClassifier

    if n_components is None:
        n_components = SVD_COMPONENTS
    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    scaler = StandardScaler()
    clf = SGDClassifier(
        loss="modified_huber",
        class_weight="balanced",
        max_iter=1_000,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    pipe = make_pipeline(svd, scaler, clf)
    pipe.fit(X_train, y_train)
    explained = svd.explained_variance_ratio_.sum()
    print(f"    SVD: {X_train.shape[1]} → {n_components} dims "
          f"({explained:.1%} variance retained)")
    return pipe.predict(X_test)


def save_fig(fig, name: str):
    path = RESULTS / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    path_png = RESULTS / f"{name}.png"
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path_png}")


# ── Q1: Which L1s benefit most from LERT? ───────────────────────────────

def q1_lert_advantage():
    """Compare LERT probe vs BERT probe per L1."""
    print("\n" + "=" * 60)
    print("Q1: Which L1s benefit most from LERT?")
    print("=" * 60)

    bert = load_report("probe_google-bert_bert-base-chinese_test")
    lert_b = load_report("probe_hfl_chinese-lert-base_test")
    lert_l = load_report("probe_hfl_chinese-lert-large_test")

    if not all([bert, lert_b, lert_l]):
        print("  Missing probe reports — skipping.")
        return

    f1_bert = per_class_f1(bert)
    f1_lert_b = per_class_f1(lert_b)
    f1_lert_l = per_class_f1(lert_l)

    # Delta: LERT-large - BERT
    common = sorted(set(f1_bert) & set(f1_lert_l))
    deltas = {lang: f1_lert_l[lang] - f1_bert[lang] for lang in common}
    deltas = dict(sorted(deltas.items(), key=lambda x: x[1], reverse=True))

    # Print table
    print(f"\n  {'L1':<15} {'BERT':>8} {'LERT-b':>8} {'LERT-L':>8} {'Δ(L-B)':>8}  Typology")
    print("  " + "-" * 62)
    for lang in deltas:
        b = f1_bert.get(lang, 0)
        lb = f1_lert_b.get(lang, 0)
        ll = f1_lert_l.get(lang, 0)
        d = deltas[lang]
        typ = "SOV" if lang in SOV_LANGUAGES else ("SVO" if lang in SVO_LANGUAGES else "—")
        print(f"  {lang:<15} {b:>8.3f} {lb:>8.3f} {ll:>8.3f} {d:>+8.3f}  {typ}")

    # Bar chart
    langs = list(deltas.keys())
    vals = [deltas[l] for l in langs]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(langs))
    ax.barh(y_pos, vals, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(langs, fontsize=9)
    ax.set_xlabel("F1 delta (LERT-large − BERT-base)")
    ax.set_title("Per-L1 advantage of LERT-large over BERT-base (probe)")
    ax.axvline(0, color="black", linewidth=0.5)
    # Annotate typology
    for i, (lang, v) in enumerate(zip(langs, vals)):
        if lang in SOV_LANGUAGES:
            ax.annotate("SOV", (v, i), fontsize=7, va="center",
                        ha="left" if v > 0 else "right",
                        xytext=(3 if v > 0 else -3, 0),
                        textcoords="offset points", color="#555")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "q1_lert_advantage")

    # Summary stats by typology
    print("\n  Average LERT-large advantage by typology:")
    for group, group_langs in [("SOV", SOV_LANGUAGES), ("SVO", SVO_LANGUAGES)]:
        ds = [deltas[l] for l in deltas if l in group_langs]
        if ds:
            print(f"    {group}: mean Δ = {np.mean(ds):+.3f}  (n={len(ds)})")

    # LaTeX table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-L1 F1 scores: LERT vs.\ BERT probes.}",
        r"\label{tab:lert-advantage}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{L1} & \textbf{Type} & \textbf{BERT} & \textbf{LERT-b} & \textbf{LERT-L} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]
    for lang in deltas:
        b = f1_bert.get(lang, 0)
        lb = f1_lert_b.get(lang, 0)
        ll = f1_lert_l.get(lang, 0)
        d = deltas[lang]
        typ = "SOV" if lang in SOV_LANGUAGES else ("SVO" if lang in SVO_LANGUAGES else "---")
        lines.append(
            f"{lang} & {typ} & {b:.3f} & {lb:.3f} & {ll:.3f} & {d:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    tex_path = RESULTS / "q1_lert_advantage.tex"
    tex_path.write_text(tex)
    print(f"  Saved {tex_path}")


# ── Q2: POS pretraining ↔ syntactic transfer ────────────────────────────

def q2_pos_correlation():
    """Train POS-only SVM, correlate per-L1 F1 with LERT advantage."""
    print("\n" + "=" * 60)
    print("Q2: Does POS-heavy pretraining correlate with syntactic signals?")
    print("=" * 60)

    from data_loader import load_and_split
    from features.pos_tags import extract_pos_features
    from features.pos_ngrams import pos_ngram_vectorizer, pos_to_sequences
    from evaluate import compute_metrics
    import scipy.sparse as sp

    # Load data
    print("  Loading data...")
    train_df, _, test_df, le = load_and_split()
    label_names = list(le.classes_)
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # POS features only (tags + n-grams)
    print("  Extracting POS-only features...")
    pos_train = extract_pos_features(train_texts)
    pos_test = extract_pos_features(test_texts)

    train_seq = pos_to_sequences(train_texts, "  POS n-gram (train)")
    test_seq = pos_to_sequences(test_texts, "  POS n-gram (test)")
    pos_vec = pos_ngram_vectorizer()
    pn_train = pos_vec.fit_transform(train_seq)
    pn_test = pos_vec.transform(test_seq)

    X_train = sp.hstack([sp.csr_matrix(pos_train), pn_train], format="csr")
    X_test = sp.hstack([sp.csr_matrix(pos_test), pn_test], format="csr")
    print(f"  POS-only feature dim: {X_train.shape[1]}")

    # Train classifier (SVD + SGD)
    print("  Training POS-only classifier...")
    y_pred = reduce_and_train(X_train, y_train, X_test)
    metrics = compute_metrics(y_test, y_pred, label_names)
    print(f"  POS-only accuracy: {metrics['accuracy']:.4f}  macro-F1: {metrics['macro_f1']:.4f}")

    # Save report
    report_path = RESULTS / "report_pos_only_test.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved {report_path}")

    # Per-L1 POS-only F1
    f1_pos = per_class_f1(metrics)

    # LERT advantage over BERT
    bert = load_report("probe_google-bert_bert-base-chinese_test")
    lert = load_report("probe_hfl_chinese-lert-large_test")
    if not bert or not lert:
        print("  Missing BERT/LERT probe reports — skipping correlation.")
        return

    f1_bert = per_class_f1(bert)
    f1_lert = per_class_f1(lert)
    common = sorted(set(f1_pos) & set(f1_bert) & set(f1_lert))

    pos_vals = [f1_pos[l] for l in common]
    lert_delta = [f1_lert[l] - f1_bert[l] for l in common]

    r, p = stats.pearsonr(pos_vals, lert_delta)
    rho, p_rho = stats.spearmanr(pos_vals, lert_delta)

    print(f"\n  Correlation: POS-only F1 vs LERT advantage")
    print(f"    Pearson  r = {r:.3f}  (p = {p:.3f})")
    print(f"    Spearman ρ = {rho:.3f}  (p = {p_rho:.3f})")
    print(f"\n  {'L1':<15} {'POS F1':>8} {'LERT Δ':>8}  Typology")
    print("  " + "-" * 42)
    for lang in common:
        typ = "SOV" if lang in SOV_LANGUAGES else ("SVO" if lang in SVO_LANGUAGES else "—")
        print(f"  {lang:<15} {f1_pos[lang]:>8.3f} {f1_lert[lang]-f1_bert[lang]:>+8.3f}  {typ}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang, x, y in zip(common, pos_vals, lert_delta):
        color = "#e74c3c" if lang in SOV_LANGUAGES else (
                "#3498db" if lang in SVO_LANGUAGES else "#95a5a6")
        ax.scatter(x, y, color=color, s=60, zorder=3)
        ax.annotate(lang, (x, y), fontsize=7, ha="left",
                    xytext=(4, 2), textcoords="offset points")

    # Regression line
    slope, intercept = np.polyfit(pos_vals, lert_delta, 1)
    xs = np.linspace(min(pos_vals) - 0.02, max(pos_vals) + 0.02, 50)
    ax.plot(xs, slope * xs + intercept, "k--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("POS-only SVM F1 (syntactic signal strength)")
    ax.set_ylabel("LERT-large advantage over BERT (ΔF1)")
    ax.set_title(f"POS signal vs LERT advantage  (r={r:.2f}, p={p:.3f})")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l)
               for c, l in [("#e74c3c", "SOV"), ("#3498db", "SVO"), ("#95a5a6", "Other")]]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "q2_pos_correlation")


# ── Q3: Are radical features implicitly captured? ────────────────────────

def q3_radical_ablation():
    """Train SVM ± radicals, compare delta with probe performance."""
    print("\n" + "=" * 60)
    print("Q3: Are radical features implicitly captured by models?")
    print("=" * 60)

    radical_map = Path(__file__).parent / "data" / "radical_map.json"
    if not radical_map.exists():
        print("  data/radical_map.json not found — run `python -m features.radicals` first.")
        return

    from data_loader import load_and_split
    from features import build_features
    from evaluate import compute_metrics

    print("  Loading data...")
    train_df, val_df, test_df, le = load_and_split()
    label_names = list(le.classes_)
    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Without radicals
    print("\n  Extracting features WITHOUT radicals...")
    X_tr_no, _, X_te_no, _ = build_features(train_texts, val_texts, test_texts,
                                             use_radicals=False)
    print("  Training classifier (no radicals)...")
    pred_no = reduce_and_train(X_tr_no, y_train, X_te_no)
    metrics_no = compute_metrics(y_test, pred_no, label_names)
    print(f"  No-radicals: acc={metrics_no['accuracy']:.4f}  F1={metrics_no['macro_f1']:.4f}")

    # With radicals
    print("\n  Extracting features WITH radicals...")
    X_tr_rad, _, X_te_rad, _ = build_features(train_texts, val_texts, test_texts,
                                               use_radicals=True)
    print("  Training classifier (with radicals)...")
    pred_rad = reduce_and_train(X_tr_rad, y_train, X_te_rad)
    metrics_rad = compute_metrics(y_test, pred_rad, label_names)
    print(f"  With radicals: acc={metrics_rad['accuracy']:.4f}  F1={metrics_rad['macro_f1']:.4f}")

    # Save reports
    with open(RESULTS / "report_svm_no_radicals_test.json", "w") as f:
        json.dump(metrics_no, f, indent=2)
    with open(RESULTS / "report_svm_with_radicals_test.json", "w") as f:
        json.dump(metrics_rad, f, indent=2)

    # Per-L1 delta from adding radicals
    f1_no = per_class_f1(metrics_no)
    f1_rad = per_class_f1(metrics_rad)

    # Best probe per L1 (max across all probes)
    probe_names = [
        "probe_google-bert_bert-base-chinese_test",
        "probe_hfl_chinese-bert-wwm_test",
        "probe_hfl_chinese-lert-base_test",
        "probe_hfl_chinese-lert-large_test",
        "probe_hfl_chinese-macbert-base_test",
        "probe_hfl_chinese-macbert-large_test",
        "probe_hfl_chinese-pert-base_test",
        "probe_hfl_chinese-roberta-wwm-ext-large_test",
        "probe_ieityuan_yuan-embedding-2.0-zh_test",
        "probe_openmoss-team_bart-base-chinese_test",
        "probe_shibing624_text2vec-base-chinese_test",
        "probe_dmetasoul_dmeta-embedding-zh-small_test",
    ]
    best_probe = {}
    for pn in probe_names:
        rep = load_report(pn)
        if not rep:
            continue
        for lang, f1 in per_class_f1(rep).items():
            if lang not in best_probe or f1 > best_probe[lang]:
                best_probe[lang] = f1

    common = sorted(set(f1_no) & set(f1_rad) & set(best_probe))
    rad_delta = {l: f1_rad[l] - f1_no[l] for l in common}

    print(f"\n  {'L1':<15} {'No rad':>8} {'+ rad':>8} {'Δ rad':>8} {'Best probe':>10}")
    print("  " + "-" * 55)
    for lang in sorted(common, key=lambda l: rad_delta[l], reverse=True):
        print(f"  {lang:<15} {f1_no[lang]:>8.3f} {f1_rad[lang]:>8.3f} "
              f"{rad_delta[lang]:>+8.3f} {best_probe[lang]:>10.3f}")

    # Correlation: radical gain vs best probe F1
    rd = [rad_delta[l] for l in common]
    bp = [best_probe[l] for l in common]
    r, p = stats.pearsonr(rd, bp)
    rho, p_rho = stats.spearmanr(rd, bp)
    print(f"\n  Correlation: radical gain vs best probe F1")
    print(f"    Pearson  r = {r:.3f}  (p = {p:.3f})")
    print(f"    Spearman ρ = {rho:.3f}  (p = {p_rho:.3f})")
    print("  (Negative r → probes already capture what radicals add)")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang, x, y in zip(common, rd, bp):
        ax.scatter(x, y, color="#2c3e50", s=60, zorder=3)
        ax.annotate(lang, (x, y), fontsize=7, ha="left",
                    xytext=(4, 2), textcoords="offset points")
    if len(common) > 2:
        slope, intercept = np.polyfit(rd, bp, 1)
        xs = np.linspace(min(rd) - 0.01, max(rd) + 0.01, 50)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("F1 gain from adding radical features (SVM)")
    ax.set_ylabel("Best probe F1 (across all encoders)")
    ax.set_title(f"Radical feature gain vs encoder probe performance  (r={r:.2f})")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    save_fig(fig, "q3_radical_ablation")

    # LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Radical feature ablation and encoder probe comparison.}",
        r"\label{tab:radical-ablation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{L1} & \textbf{SVM} & \textbf{+Rad} & \textbf{$\Delta$} & \textbf{Best Probe} \\",
        r"\midrule",
    ]
    for lang in sorted(common, key=lambda l: rad_delta[l], reverse=True):
        lines.append(
            f"{lang} & {f1_no[lang]:.3f} & {f1_rad[lang]:.3f} & "
            f"{rad_delta[lang]:+.3f} & {best_probe[lang]:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = RESULTS / "q3_radical_ablation.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved {tex_path}")


# ── Q4: Is whole-word masking helpful? ───────────────────────────────────

def q4_wwm():
    """Compare BERT (char-level masking) vs BERT-WWM (whole-word masking)."""
    print("\n" + "=" * 60)
    print("Q4: Is whole-word masking helpful for NLI?")
    print("=" * 60)

    models = {
        "BERT":       "probe_google-bert_bert-base-chinese_test",
        "BERT-WWM":   "probe_hfl_chinese-bert-wwm_test",
        "RoBERTa-WWM":"probe_hfl_chinese-roberta-wwm-ext-large_test",
        "MacBERT":    "probe_hfl_chinese-macbert-base_test",
        "LERT":       "probe_hfl_chinese-lert-base_test",
        "PERT":       "probe_hfl_chinese-pert-base_test",
    }

    reports = {}
    for label, name in models.items():
        rep = load_report(name)
        if rep:
            reports[label] = rep

    if "BERT" not in reports or "BERT-WWM" not in reports:
        print("  Missing BERT or BERT-WWM probe report — skipping.")
        return

    # Overall comparison
    print(f"\n  {'Model':<15} {'Acc':>8} {'Macro-F1':>9} {'W-F1':>8}  Masking")
    print("  " + "-" * 55)
    masking_type = {
        "BERT": "char",
        "BERT-WWM": "WWM",
        "RoBERTa-WWM": "WWM+ext",
        "MacBERT": "synonym",
        "LERT": "WWM+ling",
        "PERT": "permuted",
    }
    for label, rep in reports.items():
        acc = rep["accuracy"]
        mf1 = rep["macro_f1"]
        wf1 = rep["weighted_f1"]
        mt = masking_type.get(label, "?")
        print(f"  {label:<15} {acc:>8.3f} {mf1:>9.3f} {wf1:>8.3f}  {mt}")

    # Per-L1 comparison: BERT vs BERT-WWM
    f1_bert = per_class_f1(reports["BERT"])
    f1_wwm = per_class_f1(reports["BERT-WWM"])
    common = sorted(set(f1_bert) & set(f1_wwm))
    delta_wwm = {l: f1_wwm[l] - f1_bert[l] for l in common}

    print(f"\n  Per-L1 F1 delta (BERT-WWM − BERT):")
    print(f"  {'L1':<15} {'BERT':>8} {'WWM':>8} {'Δ':>8}")
    print("  " + "-" * 40)
    for lang in sorted(common, key=lambda l: delta_wwm[l], reverse=True):
        print(f"  {lang:<15} {f1_bert[lang]:>8.3f} {f1_wwm[lang]:>8.3f} {delta_wwm[lang]:>+8.3f}")

    avg_delta = np.mean(list(delta_wwm.values()))
    print(f"\n  Mean Δ(WWM − BERT): {avg_delta:+.3f}")

    # Grouped bar chart: all pretraining strategies per L1
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_models = [m for m in ["BERT", "BERT-WWM", "RoBERTa-WWM", "MacBERT", "LERT", "PERT"]
                  if m in reports]
    f1_per_model = {m: per_class_f1(reports[m]) for m in bar_models}
    all_langs = sorted(set().union(*(f1_per_model[m].keys() for m in bar_models)))
    # Only keep langs present in at least BERT and one other
    all_langs = [l for l in all_langs if l in f1_per_model.get("BERT", {})]

    n_models = len(bar_models)
    n_langs = len(all_langs)
    width = 0.8 / n_models
    x = np.arange(n_langs)
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#95a5a6"]

    for i, model in enumerate(bar_models):
        vals = [f1_per_model[model].get(l, 0) for l in all_langs]
        ax.bar(x + i * width, vals, width, label=f"{model} ({masking_type[model]})",
               color=colors[i % len(colors)], edgecolor="none")

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(all_langs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-L1 F1 by pretraining strategy (probes)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, "q4_masking_comparison")

    # LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Probe F1 by pretraining strategy.}",
        r"\label{tab:masking-comparison}",
        r"\begin{tabular}{l" + "r" * n_models + "}",
        r"\toprule",
        r"\textbf{L1} & " + " & ".join(
            rf"\textbf{{{m}}}" for m in bar_models) + r" \\",
        r"\midrule",
    ]
    for lang in all_langs:
        vals = " & ".join(f"{f1_per_model[m].get(lang, 0):.3f}" for m in bar_models)
        lines.append(f"{lang} & {vals} \\\\")
    # Macro avg row
    lines.append(r"\midrule")
    macro_vals = " & ".join(f"{reports[m]['macro_f1']:.3f}" for m in bar_models)
    lines.append(f"\\textit{{Macro avg}} & {macro_vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = RESULTS / "q4_masking_comparison.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved {tex_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses (including SVM training)")
    parser.add_argument("--pos-only", action="store_true",
                        help="Run Q2 (POS correlation, trains a POS-only SVM)")
    parser.add_argument("--radicals", action="store_true",
                        help="Run Q3 (radical ablation, trains SVM twice)")
    parser.add_argument("--svd", type=int, default=300, metavar="N",
                        help="TruncatedSVD components (default: 300)")
    args = parser.parse_args()

    import analysis_discussion
    analysis_discussion.SVD_COMPONENTS = args.svd

    # Q1 and Q4 are fast (JSON only)
    q1_lert_advantage()
    q4_wwm()

    if args.all or args.pos_only:
        q2_pos_correlation()

    if args.all or args.radicals:
        q3_radical_ablation()

    if not args.all and not args.pos_only and not args.radicals:
        print("\nNote: Q2 and Q3 require training SVMs. Run with --all, --pos-only, or --radicals.")

    print("\nDone.")


if __name__ == "__main__":
    main()
