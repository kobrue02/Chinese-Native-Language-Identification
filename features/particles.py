"""Particle context features for Chinese NLI.

The misuse of grammatical particles (了, 的, 把, 被, 着, 过, 在, 得) is
among the strongest NLI discriminators in Chinese. This module captures
not just whether a particle is used, but *how* — by extracting the POS
tags of words surrounding each particle occurrence.

For example, a Korean speaker might produce "VERB 了 了" (double 了),
while a Thai speaker might omit 了 after accomplishment verbs.
"""

import numpy as np
import jieba.posseg as pseg
from tqdm import tqdm


# Key grammatical particles and their role
PARTICLES = ["了", "的", "把", "被", "着", "过", "在", "得", "地"]

# Context window: POS of (prev word, particle, next word)
# We build features for each particle × each context position × common POS tags
_CONTEXT_POS = [
    "n", "v", "a", "d", "r", "p", "m", "q", "c", "u", "x", "OTHER",
]


def _pos_index(tag: str) -> int:
    """Map a POS tag to our reduced index set."""
    for i, t in enumerate(_CONTEXT_POS[:-1]):
        if tag.startswith(t):
            return i
    return len(_CONTEXT_POS) - 1  # OTHER


def extract_particle_features(texts: list[str]) -> np.ndarray:
    """Extract particle context features.

    For each of the 9 particles, we extract:
      - Raw frequency (normalized by text length)                    → 1
      - POS of preceding word (12 bins)                              → 12
      - POS of following word (12 bins)                              → 12
      - Whether particle appears at sentence start/end               → 2
    Total: 9 × 27 = 243 features
    """
    n_pos = len(_CONTEXT_POS)
    feats_per_particle = 1 + n_pos + n_pos + 2  # 27
    n_features = len(PARTICLES) * feats_per_particle
    features = np.zeros((len(texts), n_features), dtype=np.float64)

    for i, text in enumerate(tqdm(texts, desc="Particle context")):
        words_and_flags = list(pseg.cut(text))
        n_tokens = len(words_and_flags) or 1

        for p_idx, particle in enumerate(PARTICLES):
            base = p_idx * feats_per_particle

            for j, (word, flag) in enumerate(words_and_flags):
                if word != particle:
                    continue

                # Raw count
                features[i, base] += 1

                # POS of previous word
                if j > 0:
                    prev_pos = _pos_index(words_and_flags[j - 1][1])
                    features[i, base + 1 + prev_pos] += 1

                # POS of next word
                if j < len(words_and_flags) - 1:
                    next_pos = _pos_index(words_and_flags[j + 1][1])
                    features[i, base + 1 + n_pos + next_pos] += 1

                # Sentence boundary indicators
                if j == 0 or words_and_flags[j - 1][0] in "。！？!?":
                    features[i, base + 1 + 2 * n_pos] += 1  # at start
                if j == len(words_and_flags) - 1 or (
                    j < len(words_and_flags) - 1
                    and words_and_flags[j + 1][0] in "。！？!?"
                ):
                    features[i, base + 1 + 2 * n_pos + 1] += 1  # at end

            # Normalize frequency by text length
            features[i, base] /= n_tokens

    return features
