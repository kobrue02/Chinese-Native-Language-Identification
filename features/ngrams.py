"""Character and word n-gram TF-IDF feature extraction."""

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

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


def _jieba_tokenizer(text: str) -> list[str]:
    """Tokenize Chinese text with jieba."""
    return list(jieba.cut(text))


def word_ngram_vectorizer(
    ngram_range: tuple[int, int] | None = None,
    max_features: int | None = None,
) -> TfidfVectorizer:
    """Return a TF-IDF vectorizer for word n-grams (jieba segmentation)."""
    ngram_range = ngram_range or config.SVM_CONFIG["word_ngram_range"]
    max_features = max_features or config.SVM_CONFIG["max_features"]
    return TfidfVectorizer(
        tokenizer=_jieba_tokenizer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=None,  # suppress warning when using custom tokenizer
    )
