"""Dependency triple features via spaCy.

Triples of (head POS, dep label, child POS) capture syntactic structure
without lexical content — e.g., (VERB, nsubj, NOUN) patterns that
differ across L1 backgrounds.

Requires: python -m spacy download zh_core_web_sm
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

_NLP = None


def _get_nlp():
    """Lazy-load spaCy model."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("zh_core_web_sm")
    return _NLP


def _text_to_dep_triples(text: str) -> str:
    """Extract dependency triples as a space-separated string of tokens.

    Each triple is encoded as 'headPOS_deprel_childPOS'.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    triples = []
    for token in doc:
        if token.dep_ != "ROOT":
            triple = f"{token.head.pos_}_{token.dep_}_{token.pos_}"
            triples.append(triple)
    return " ".join(triples)


def texts_to_dep_triples(
    texts: list[str], desc: str = "Dep parsing"
) -> list[str]:
    """Convert texts to dependency triple sequences (batched via spaCy pipe)."""
    nlp = _get_nlp()
    results = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=64, n_process=1),
        total=len(texts),
        desc=desc,
    ):
        triples = []
        for token in doc:
            if token.dep_ != "ROOT":
                triples.append(f"{token.head.pos_}_{token.dep_}_{token.pos_}")
        results.append(" ".join(triples))
    return results


def dep_triple_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 20_000,
) -> TfidfVectorizer:
    """TF-IDF vectorizer over dependency triples."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
