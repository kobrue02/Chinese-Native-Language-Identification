"""Lexical richness and surface-level features.

Captures vocabulary diversity, punctuation habits, and character-level
statistics that often differ across L1 backgrounds.
"""

import re

import numpy as np
from tqdm import tqdm

# Chinese punctuation (full-width)
_ZH_PUNCT = set("，。！？、；：""''（）【】《》—…·")
# ASCII punctuation
_EN_PUNCT = set(",.!?;:'\"()-")


def extract_lexical_richness_features(texts: list[str]) -> np.ndarray:
    """Extract lexical richness and surface features.

    Features per document (13 total):
      0: type_token_ratio       - unique words / total words
      1: char_type_token_ratio  - unique chars / total chars
      2: hapax_ratio            - words appearing once / total words
      3: avg_word_len           - average word length in characters
      4: text_length_chars      - total character count
      5: text_length_words      - total word count
      6: zh_punct_ratio         - Chinese punctuation / total chars
      7: en_punct_ratio         - ASCII punctuation / total chars
      8: mixed_punct_ratio      - using both zh + en punct in same text
      9: digit_ratio            - digits / total chars
     10: latin_ratio            - Latin letters / total chars
     11: unique_char_count      - number of distinct characters
     12: avg_sentence_len_words - average sentence length in words
    """
    import jieba

    n_features = 13
    features = np.zeros((len(texts), n_features), dtype=np.float64)

    from collections import Counter

    for i, text in enumerate(tqdm(texts, desc="Lexical richness")):
        if not text.strip():
            continue  # leave zeros for empty texts

        words = list(jieba.cut(text))
        chars = list(text)
        n_words = len(words) or 1
        n_chars = len(chars) or 1

        # Type-token ratios
        word_types = set(words)
        features[i, 0] = len(word_types) / n_words
        features[i, 1] = len(set(chars)) / n_chars

        # Hapax legomena (words appearing exactly once)
        word_counts = Counter(words)
        hapax = sum(1 for c in word_counts.values() if c == 1)
        features[i, 2] = hapax / n_words

        # Average word length
        features[i, 3] = np.mean([len(w) for w in words]) if words else 0.0

        # Text length
        features[i, 4] = n_chars
        features[i, 5] = n_words

        # Punctuation ratios
        zh_punct = sum(1 for c in chars if c in _ZH_PUNCT)
        en_punct = sum(1 for c in chars if c in _EN_PUNCT)
        features[i, 6] = zh_punct / n_chars
        features[i, 7] = en_punct / n_chars
        features[i, 8] = 1.0 if (zh_punct > 0 and en_punct > 0) else 0.0

        # Digit and Latin letter ratios
        features[i, 9] = sum(1 for c in chars if c.isdigit()) / n_chars
        features[i, 10] = sum(1 for c in chars if c.isascii() and c.isalpha()) / n_chars

        # Unique character count
        features[i, 11] = len(set(chars))

        # Average sentence length in words
        sentences = re.split(r"[。！？!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        if sentences:
            sent_word_lens = [len(list(jieba.cut(s))) for s in sentences]
            features[i, 12] = np.mean(sent_word_lens) if sent_word_lens else 0.0

    return features
