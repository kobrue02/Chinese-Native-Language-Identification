"""POS tag n-gram features (de-lexicalized syntax).

POS bigrams and trigrams capture shallow syntactic patterns without
topic bias — e.g., frequent DET-ADJ-NOUN sequences revealing L1
determiner transfer.
"""

import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def _text_to_pos_sequence(text: str) -> str:
    """Convert text to a space-separated sequence of POS tags."""
    return " ".join(flag for _, flag in pseg.cut(text))


def pos_to_sequences(texts: list[str], desc: str = "POS tagging") -> list[str]:
    """Convert a list of texts to POS tag sequences."""
    return [_text_to_pos_sequence(t) for t in tqdm(texts, desc=desc)]


def pos_ngram_vectorizer(
    ngram_range: tuple[int, int] = (2, 3),
    max_features: int = 20_000,
) -> TfidfVectorizer:
    """TF-IDF vectorizer over POS tag n-grams."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
