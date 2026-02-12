"""Segmentation-derived features for Chinese NLI.

Since Chinese has no word boundaries, the way a standard segmenter
(jieba) tokenizes learner text is itself a signal. L1-specific errors
and word choices lead to characteristic segmentation patterns — e.g.,
more single-character "words" for learners who haven't acquired
multi-character compounds.
"""

import numpy as np
import jieba
from collections import Counter
from tqdm import tqdm


def extract_segmentation_features(texts: list[str]) -> np.ndarray:
    """Extract features derived from jieba word segmentation.

    Features per document (10 total):
      0: avg_word_length        - average word length in characters
      1: single_char_ratio      - proportion of single-character words
      2: two_char_ratio         - proportion of two-character words
      3: three_plus_char_ratio  - proportion of 3+-character words
      4: vocab_size             - number of unique words
      5: oov_ratio              - words appearing only once / total words
      6: max_word_length        - longest word in characters
      7: word_length_std        - standard deviation of word lengths
      8: segmentation_density   - words per character (higher = more segmentation)
      9: compound_ratio         - ratio of multi-char words to single-char words
    """
    n_features = 10
    features = np.zeros((len(texts), n_features), dtype=np.float64)

    for i, text in enumerate(tqdm(texts, desc="Segmentation features")):
        if not text.strip():
            continue

        words = list(jieba.cut(text))
        if not words:
            continue

        n_words = len(words)
        word_lens = [len(w) for w in words]
        n_chars = len(text) or 1

        # Average word length
        features[i, 0] = np.mean(word_lens)

        # Word length distribution
        single = sum(1 for l in word_lens if l == 1)
        two = sum(1 for l in word_lens if l == 2)
        three_plus = sum(1 for l in word_lens if l >= 3)
        features[i, 1] = single / n_words
        features[i, 2] = two / n_words
        features[i, 3] = three_plus / n_words

        # Vocabulary size
        features[i, 4] = len(set(words))

        # OOV ratio (hapax in segmented text)
        word_counts = Counter(words)
        features[i, 5] = sum(1 for c in word_counts.values() if c == 1) / n_words

        # Max word length
        features[i, 6] = max(word_lens)

        # Word length standard deviation
        features[i, 7] = np.std(word_lens) if len(word_lens) > 1 else 0.0

        # Segmentation density
        features[i, 8] = n_words / n_chars

        # Compound ratio (multi-char / single-char)
        features[i, 9] = (n_words - single) / max(single, 1)

    return features
