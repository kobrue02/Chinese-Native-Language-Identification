import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm

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
    tqdm.pandas(desc="Counting words (jieba)")
    df["word_len"] = df["text"].progress_apply(lambda t: len(list(jieba.cut(t))))

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


# ── LaTeX table exports ───────────────────────────────────────────────────


def _save_tex(content: str, name: str) -> None:
    path = config.RESULTS_DIR / f"{name}.tex"
    path.write_text(content)
    print(f"Saved {name}.tex")


def export_corpus_overview_tex(df: pd.DataFrame) -> None:
    """Export a compact corpus overview table."""
    n_docs = len(df)
    n_langs = df["native_language"].nunique()
    n_contexts = df["context"].nunique()
    char_lens = df["text"].str.len()

    rows = [
        ("Documents", f"{n_docs:,}"),
        ("Native languages", str(n_langs)),
        ("Writing contexts", str(n_contexts)),
        ("Mean text length (chars)", f"{char_lens.mean():.0f}"),
        ("Median text length (chars)", f"{char_lens.median():.0f}"),
        ("Min / Max text length", f"{char_lens.min()} / {char_lens.max():,}"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Corpus overview.}",
        r"\label{tab:corpus-overview}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Statistic} & \textbf{Value} \\",
        r"\midrule",
    ]
    for label, val in rows:
        lines.append(f"{label} & {val} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_tex("\n".join(lines), "corpus_overview")


def export_language_distribution_tex(df: pd.DataFrame) -> None:
    """Export language distribution as a LaTeX table (sorted by count)."""
    counts = df["native_language"].value_counts()
    total = counts.sum()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sample counts per native language.}",
        r"\label{tab:language-distribution}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Native Language} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]
    for lang, cnt in counts.items():
        pct = 100 * cnt / total
        lines.append(f"{lang} & {cnt} & {pct:.1f} \\\\")
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100.0}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _save_tex("\n".join(lines), "language_distribution")


def export_context_distribution_tex(df: pd.DataFrame) -> None:
    """Export writing context distribution as a LaTeX table."""
    counts = df["context"].value_counts()
    total = counts.sum()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sample counts per writing context.}",
        r"\label{tab:context-distribution}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Context} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]
    for ctx, cnt in counts.items():
        pct = 100 * cnt / total
        lines.append(f"{ctx} & {cnt} & {pct:.1f} \\\\")
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{100.0}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    _save_tex("\n".join(lines), "context_distribution")


def export_language_context_crosstab_tex(df: pd.DataFrame) -> None:
    """Export language × context cross-tab as a LaTeX table."""
    ct = pd.crosstab(df["native_language"], df["context"], margins=True)
    ct = ct.sort_values("All", ascending=False)

    contexts = [c for c in ct.columns if c != "All"]
    col_header = " & ".join([r"\textbf{" + c + "}" for c in contexts])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Native language $\times$ writing context.}",
        r"\label{tab:language-context}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(contexts) + "r}",
        r"\toprule",
        r"\textbf{Language} & " + col_header + r" & \textbf{Total} \\",
        r"\midrule",
    ]
    for lang in ct.index:
        vals = " & ".join([str(ct.loc[lang, c]) for c in contexts])
        total = ct.loc[lang, "All"]
        bold = lang == "All"
        if bold:
            lines.append(r"\midrule")
            lines.append(
                r"\textbf{Total} & "
                + " & ".join([r"\textbf{" + str(ct.loc[lang, c]) + "}" for c in contexts])
                + r" & \textbf{"
                + str(total)
                + r"} \\"
            )
        else:
            lines.append(f"{lang} & {vals} & {total} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    _save_tex("\n".join(lines), "language_context_crosstab")


def export_text_length_stats_tex(df: pd.DataFrame) -> None:
    """Export per-language text length statistics."""
    df = df.copy()
    df["char_len"] = df["text"].str.len()

    stats = (
        df.groupby("native_language")["char_len"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .sort_values("count", ascending=False)
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Text length statistics (characters) per native language.}",
        r"\label{tab:text-length-stats}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Language} & \textbf{N} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Median} & \textbf{Max} \\",
        r"\midrule",
    ]
    for lang, row in stats.iterrows():
        lines.append(
            f"{lang} & {row['count']:.0f} & {row['mean']:.0f} & {row['std']:.0f} "
            f"& {row['min']:.0f} & {row['median']:.0f} & {row['max']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _save_tex("\n".join(lines), "text_length_stats")


# ── TikZ / pgfplots exports ───────────────────────────────────────────────


def _save_dat(xs, ys, name: str) -> None:
    """Save (x, y) data as a space-separated .dat file for pgfplots."""
    path = config.RESULTS_DIR / f"{name}.dat"
    with open(path, "w") as f:
        f.write("x y\n")
        for x, y in zip(xs, ys):
            f.write(f"{x:.2f} {y:.6f}\n")


def export_length_distribution_tikz(df: pd.DataFrame) -> None:
    """Export text length KDE as a TikZ/pgfplots line chart.

    Produces two overlaid density curves (characters and words).
    """
    df = df.copy()
    char_lens = df["text"].str.len().values.astype(float)
    tqdm.pandas(desc="Word lengths for TikZ (jieba)")
    word_lens = df["text"].progress_apply(lambda t: len(list(jieba.cut(t)))).values.astype(float)

    # Compute KDE for both
    for data, label in [(char_lens, "chars"), (word_lens, "words")]:
        kde = gaussian_kde(data, bw_method="scott")
        x_min, x_max = data.min(), np.percentile(data, 99)
        xs = np.linspace(x_min, x_max, 200)
        ys = kde(xs)
        _save_dat(xs, ys, f"length_kde_{label}")

    char_max = np.percentile(char_lens, 99)
    word_max = np.percentile(word_lens, 99)

    tikz = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.9\textwidth,
    height=6cm,
    xlabel={{Text length}},
    ylabel={{Density}},
    legend style={{at={{(0.97,0.97)}}, anchor=north east}},
    grid=major,
    grid style={{gray!30}},
    no markers,
    thick,
]
\addplot[blue, solid] table[x=x, y=y] {{length_kde_chars.dat}};
\addlegendentry{{Characters}}
\addplot[red, dashed] table[x=x, y=y] {{length_kde_words.dat}};
\addlegendentry{{Words (jieba)}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Text length distribution (kernel density estimate).}}
\label{{fig:length-distribution}}
\end{{figure}}"""
    _save_tex(tikz, "length_distribution_tikz")
    print("Saved length_kde_chars.dat, length_kde_words.dat")


def export_length_distribution_by_language_tikz(
    df: pd.DataFrame, top_n: int = 5
) -> None:
    """Export per-language text length KDE for the top N languages."""
    top_langs = df["native_language"].value_counts().head(top_n).index.tolist()
    colors = ["blue", "red", "green!60!black", "orange", "purple"]

    plot_cmds = []
    for lang, color in zip(top_langs, colors):
        data = df.loc[df["native_language"] == lang, "text"].str.len().values.astype(float)
        kde = gaussian_kde(data, bw_method="scott")
        x_max = np.percentile(data, 99)
        xs = np.linspace(data.min(), x_max, 200)
        ys = kde(xs)
        safe_name = lang.lower().replace(" ", "_")
        _save_dat(xs, ys, f"length_kde_lang_{safe_name}")
        plot_cmds.append(
            rf"\addplot[{color}, thick] table[x=x, y=y] {{length_kde_lang_{safe_name}.dat}};"
            f"\n\\addlegendentry{{{lang}}}"
        )

    plots = "\n".join(plot_cmds)
    tikz = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.9\textwidth,
    height=6cm,
    xlabel={{Text length (characters)}},
    ylabel={{Density}},
    legend style={{at={{(0.97,0.97)}}, anchor=north east, font=\small}},
    grid=major,
    grid style={{gray!30}},
    no markers,
]
{plots}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Text length distribution by native language (top {top_n}).}}
\label{{fig:length-by-language}}
\end{{figure}}"""
    _save_tex(tikz, "length_by_language_tikz")


def export_language_distribution_tikz(df: pd.DataFrame, top_n: int = 15) -> None:
    """Export horizontal bar chart of language distribution as TikZ/pgfplots."""
    counts = df["native_language"].value_counts()
    top = counts.head(top_n)
    other = counts.iloc[top_n:].sum()
    if other > 0:
        top = pd.concat([top, pd.Series({"Other": other})])

    # Reverse so largest is at top in horizontal bar chart
    top = top.iloc[::-1]

    # Save data
    dat_path = config.RESULTS_DIR / "language_counts.dat"
    with open(dat_path, "w") as f:
        f.write("idx language count\n")
        for i, (lang, cnt) in enumerate(top.items()):
            f.write(f"{i} {{{lang}}} {cnt}\n")

    yticklabels = ", ".join(f"{{{lang}}}" for lang in top.index)
    n = len(top)

    tikz = rf"""\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.85\textwidth,
    height={max(5, n * 0.45):.1f}cm,
    xbar,
    xlabel={{Count}},
    ytick={{{",".join(str(i) for i in range(n))}}},
    yticklabels={{{yticklabels}}},
    y dir=normal,
    bar width=8pt,
    xmin=0,
    enlarge y limits={{abs=0.4}},
    nodes near coords,
    nodes near coords align={{horizontal}},
    every node near coord/.append style={{font=\scriptsize}},
]
\addplot[fill=blue!60, draw=blue!80] table[x=count, y=idx] {{language_counts.dat}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{Sample counts per native language (top {top_n}{" + Other" if other > 0 else ""}).}}
\label{{fig:language-distribution}}
\end{{figure}}"""
    _save_tex(tikz, "language_distribution_tikz")
    print("Saved language_counts.dat")


if __name__ == "__main__":
    print("Loading corpus...")
    df = load_corpus()
    print_summary_stats(df)
    print("\nGenerating plots...")
    plot_language_distribution(df)
    plot_context_distribution(df)
    plot_text_lengths(df)
    plot_language_context_heatmap(df)
    print("\nExporting LaTeX tables...")
    export_corpus_overview_tex(df)
    export_language_distribution_tex(df)
    export_context_distribution_tex(df)
    export_language_context_crosstab_tex(df)
    export_text_length_stats_tex(df)
    print("\nExporting TikZ plots...")
    export_length_distribution_tikz(df)
    export_length_distribution_by_language_tikz(df)
    export_language_distribution_tikz(df)
    print("\nAll outputs saved to", config.RESULTS_DIR)
