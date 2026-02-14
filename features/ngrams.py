"""Character and word n-gram TF-IDF feature extraction."""

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import config


def char_ngram_vectorizer(
    ngram_range: tuple[int, int] | None = None,
    max_features: int | None = None,
) -> TfidfVectorizer:
    """Return a TF-IDF vectorizer for character n-grams."""
    ngram_range = ngram_range or config.SVM_CONFIG["char_ngram_range"]
    max_features = max_features or config.SVM_CONFIG["max_features"]
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )


class _JiebaTokenizerWithProgress:
    """Jieba tokenizer that updates a shared tqdm progress bar."""

    def __init__(self):
        self._pbar = None

    def set_total(self, n: int, desc: str = "Jieba tokenizing"):
        """Create a fresh progress bar for n documents."""
        if self._pbar is not None:
            self._pbar.close()
        self._pbar = tqdm(total=n, desc=desc)

    def __call__(self, text: str) -> list[str]:
        tokens = list(jieba.cut(text))
        if self._pbar is not None:
            self._pbar.update(1)
        return tokens

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


def word_ngram_vectorizer(
    ngram_range: tuple[int, int] | None = None,
    max_features: int | None = None,
) -> tuple[TfidfVectorizer, _JiebaTokenizerWithProgress]:
    """Return a TF-IDF vectorizer for word n-grams (jieba segmentation).

    Also returns the tokenizer object so callers can call
    `tokenizer.set_total(n)` before fit/transform.
    """
    ngram_range = ngram_range or config.SVM_CONFIG["word_ngram_range"]
    max_features = max_features or config.SVM_CONFIG["max_features"]
    tokenizer = _JiebaTokenizerWithProgress()
    vec = TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=None,
    )
    return vec, tokenizer
