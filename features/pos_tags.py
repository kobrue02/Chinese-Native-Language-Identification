"""POS tag distribution features via jieba.posseg."""

import numpy as np
import jieba.posseg as pseg
from sklearn.preprocessing import normalize

# All jieba POS tags we track (ICTCLAS tagset)
POS_TAGS = [
    "a",
    "ad",
    "ag",
    "an",  # adjectives
    "b",
    "c",
    "d",
    "dg",  # distinguishing, conjunction, adverb
    "e",
    "f",
    "g",
    "h",
    "i",  # interjection, position, morpheme, prefix, idiom
    "j",
    "k",
    "l",
    "m",
    "mg",  # abbreviation, suffix, temp, numeral
    "n",
    "ng",
    "nr",
    "nrfg",
    "nrt",
    "ns",
    "nt",
    "nz",  # nouns
    "o",
    "p",
    "q",
    "r",
    "rg",
    "rr",
    "rz",  # onomatopoeia, prep, measure, pronoun
    "s",
    "t",
    "tg",
    "u",
    "ud",
    "ug",
    "uj",
    "ul",
    "uv",
    "uz",  # space, time, aux
    "v",
    "vd",
    "vg",
    "vi",
    "vn",
    "vq",  # verbs
    "x",
    "y",
    "z",
    "zg",
    "eng",  # non-morpheme, modal, descriptive, other
]

_TAG2IDX = {tag: i for i, tag in enumerate(POS_TAGS)}


def extract_pos_features(texts: list[str]) -> np.ndarray:
    """Extract POS tag frequency vectors for a list of texts.

    Returns an (n_docs, n_tags) matrix, L1-normalized per document.
    """
    n = len(texts)
    features = np.zeros((n, len(POS_TAGS)), dtype=np.float64)

    for i, text in enumerate(texts):
        for word, flag in pseg.cut(text):
            idx = _TAG2IDX.get(flag)
            if idx is not None:
                features[i, idx] += 1

    # L1-normalize so each row sums to 1 (frequency distribution)
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    features = features / row_sums

    return features
