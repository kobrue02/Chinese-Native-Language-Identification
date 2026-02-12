"""Discourse and pragmatic features.

- Discourse connective frequencies: overuse/underuse of transition words
  reflects L1 rhetorical norms.
- Sentence position features: how sentences typically start/end reveals
  L1 writing conventions.
"""

import re

import numpy as np
from tqdm import tqdm

# Chinese discourse connectives grouped by function
CONNECTIVES = {
    # Additive
    "而且": 0, "并且": 1, "同时": 2, "另外": 3, "此外": 4, "还有": 5,
    # Adversative
    "但是": 6, "可是": 7, "不过": 8, "然而": 9, "却": 10, "倒": 11,
    # Causal
    "因为": 12, "所以": 13, "因此": 14, "由于": 15, "以致": 16,
    # Conditional
    "如果": 17, "假如": 18, "要是": 19, "只要": 20, "除非": 21,
    # Concessive
    "虽然": 22, "尽管": 23, "即使": 24, "哪怕": 25,
    # Sequential / temporal
    "首先": 26, "然后": 27, "接着": 28, "最后": 29, "终于": 30,
    "以前": 31, "以后": 32, "之后": 33, "之前": 34,
    # Exemplifying
    "例如": 35, "比如": 36, "譬如": 37,
    # Summarizing
    "总之": 38, "总的来说": 39, "综上所述": 40, "总而言之": 41,
}

N_CONNECTIVES = len(CONNECTIVES)

# Sentence-splitting pattern: Chinese period, question mark, exclamation
_SENT_SPLIT = re.compile(r"[。！？!?]+")


def extract_discourse_features(texts: list[str]) -> np.ndarray:
    """Extract discourse connective frequencies + sentence-level features.

    Returns (n_docs, n_connectives + 4) matrix:
      - First n_connectives columns: normalized connective counts
      - avg_sent_len: average sentence length in characters
      - n_sentences: total number of sentences
      - first_word_variety: unique first characters across sentences / n_sentences
      - last_word_variety: unique last characters across sentences / n_sentences
    """
    n_extra = 4
    features = np.zeros((len(texts), N_CONNECTIVES + n_extra), dtype=np.float64)

    for i, text in enumerate(tqdm(texts, desc="Discourse features")):
        # Connective counts
        for conn, idx in CONNECTIVES.items():
            features[i, idx] = text.count(conn)
        total_chars = len(text) or 1
        features[i, :N_CONNECTIVES] /= total_chars  # normalize

        # Sentence-level features
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        n_sent = len(sentences) or 1

        avg_len = np.mean([len(s) for s in sentences]) if sentences else 0
        features[i, N_CONNECTIVES] = avg_len

        features[i, N_CONNECTIVES + 1] = n_sent

        # Variety of sentence-initial / sentence-final characters
        first_chars = set(s[0] for s in sentences if s)
        last_chars = set(s[-1] for s in sentences if s)
        features[i, N_CONNECTIVES + 2] = len(first_chars) / n_sent
        features[i, N_CONNECTIVES + 3] = len(last_chars) / n_sent

    return features
