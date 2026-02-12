"""Feature extraction: combine n-gram TF-IDF and POS tag features."""

import numpy as np
import scipy.sparse as sp

from features.ngrams import char_ngram_vectorizer, word_ngram_vectorizer
from features.pos_tags import extract_pos_features


def build_features(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, dict]:
    """Build combined feature matrices for train/val/test.

    Returns (X_train, X_val, X_test, vectorizers_dict).
    """
    # Character n-grams
    char_vec = char_ngram_vectorizer()
    X_train_char = char_vec.fit_transform(train_texts)
    X_val_char = char_vec.transform(val_texts)
    X_test_char = char_vec.transform(test_texts)

    # Word n-grams
    word_vec = word_ngram_vectorizer()
    X_train_word = word_vec.fit_transform(train_texts)
    X_val_word = word_vec.transform(val_texts)
    X_test_word = word_vec.transform(test_texts)

    # POS tag features
    X_train_pos = sp.csr_matrix(extract_pos_features(train_texts))
    X_val_pos = sp.csr_matrix(extract_pos_features(val_texts))
    X_test_pos = sp.csr_matrix(extract_pos_features(test_texts))

    # Combine all
    X_train = sp.hstack([X_train_char, X_train_word, X_train_pos], format="csr")
    X_val = sp.hstack([X_val_char, X_val_word, X_val_pos], format="csr")
    X_test = sp.hstack([X_test_char, X_test_word, X_test_pos], format="csr")

    vectorizers = {"char": char_vec, "word": word_vec}

    return X_train, X_val, X_test, vectorizers
