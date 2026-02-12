<<<<<<< HEAD
"""Feature extraction: combine n-gram TF-IDF and POS tag features."""
=======
"""Feature extraction: combine all feature types for SVM."""
>>>>>>> e689c55 (add more scripts and features)

import numpy as np
import scipy.sparse as sp

from features.ngrams import char_ngram_vectorizer, word_ngram_vectorizer
from features.pos_tags import extract_pos_features
<<<<<<< HEAD
=======
from features.pos_ngrams import pos_ngram_vectorizer, pos_to_sequences
from features.function_words import extract_function_word_features
from features.discourse import extract_discourse_features
from features.lexical_richness import extract_lexical_richness_features
from features.particles import extract_particle_features
from features.segmentation import extract_segmentation_features


def _sparse(arr: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(arr)
>>>>>>> e689c55 (add more scripts and features)


def build_features(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
<<<<<<< HEAD
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
=======
    use_dependency: bool = False,
    use_radicals: bool = False,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, dict]:
    """Build combined feature matrices for train/val/test.

    Args:
        use_dependency: Include spaCy dependency triples (requires zh_core_web_sm).
        use_radicals: Include radical features (requires data/radical_map.json;
            generate with ``python -m features.radicals``).

    Returns (X_train, X_val, X_test, vectorizers_dict).
    """
    blocks_train, blocks_val, blocks_test = [], [], []
    vectorizers = {}
    step = 0
    total = 9 + 2 * int(use_radicals) + int(use_dependency)

    def header(name: str):
        nonlocal step
        step += 1
        print(f"  [{step}/{total}] {name}")

    # ── 1. Character n-grams ──────────────────────────────────────────────
    header("Character n-gram TF-IDF...")
    char_vec = char_ngram_vectorizer()
    blocks_train.append(char_vec.fit_transform(train_texts))
    blocks_val.append(char_vec.transform(val_texts))
    blocks_test.append(char_vec.transform(test_texts))
    vectorizers["char"] = char_vec

    # ── 2. Word n-grams (jieba) ───────────────────────────────────────────
    header("Word n-gram TF-IDF (jieba)...")
    word_vec, jieba_tok = word_ngram_vectorizer()
    jieba_tok.set_total(len(train_texts), desc="    fit (train)")
    blocks_train.append(word_vec.fit_transform(train_texts))
    jieba_tok.close()
    jieba_tok.set_total(len(val_texts), desc="    transform (val)")
    blocks_val.append(word_vec.transform(val_texts))
    jieba_tok.close()
    jieba_tok.set_total(len(test_texts), desc="    transform (test)")
    blocks_test.append(word_vec.transform(test_texts))
    jieba_tok.close()
    vectorizers["word"] = word_vec

    # ── 3. POS tag distribution ───────────────────────────────────────────
    header("POS tag distributions...")
    blocks_train.append(_sparse(extract_pos_features(train_texts)))
    blocks_val.append(_sparse(extract_pos_features(val_texts)))
    blocks_test.append(_sparse(extract_pos_features(test_texts)))

    # ── 4. POS n-grams (de-lexicalized syntax) ───────────────────────────
    header("POS n-gram TF-IDF...")
    train_pos_seq = pos_to_sequences(train_texts, desc="    POS seq (train)")
    val_pos_seq = pos_to_sequences(val_texts, desc="    POS seq (val)")
    test_pos_seq = pos_to_sequences(test_texts, desc="    POS seq (test)")
    pos_vec = pos_ngram_vectorizer()
    blocks_train.append(pos_vec.fit_transform(train_pos_seq))
    blocks_val.append(pos_vec.transform(val_pos_seq))
    blocks_test.append(pos_vec.transform(test_pos_seq))
    vectorizers["pos_ngram"] = pos_vec

    # ── 5. Function word frequencies ──────────────────────────────────────
    header("Function word frequencies...")
    blocks_train.append(_sparse(extract_function_word_features(train_texts)))
    blocks_val.append(_sparse(extract_function_word_features(val_texts)))
    blocks_test.append(_sparse(extract_function_word_features(test_texts)))

    # ── 6. Particle context patterns ──────────────────────────────────────
    header("Particle context features (了/的/把/被/着/过/在/得/地)...")
    blocks_train.append(_sparse(extract_particle_features(train_texts)))
    blocks_val.append(_sparse(extract_particle_features(val_texts)))
    blocks_test.append(_sparse(extract_particle_features(test_texts)))

    # ── 7. Discourse & sentence-level features ────────────────────────────
    header("Discourse connectives & sentence features...")
    blocks_train.append(_sparse(extract_discourse_features(train_texts)))
    blocks_val.append(_sparse(extract_discourse_features(val_texts)))
    blocks_test.append(_sparse(extract_discourse_features(test_texts)))

    # ── 8. Lexical richness ───────────────────────────────────────────────
    header("Lexical richness features...")
    blocks_train.append(_sparse(extract_lexical_richness_features(train_texts)))
    blocks_val.append(_sparse(extract_lexical_richness_features(val_texts)))
    blocks_test.append(_sparse(extract_lexical_richness_features(test_texts)))

    # ── 9. Segmentation-derived features ──────────────────────────────────
    header("Segmentation-derived features...")
    blocks_train.append(_sparse(extract_segmentation_features(train_texts)))
    blocks_val.append(_sparse(extract_segmentation_features(val_texts)))
    blocks_test.append(_sparse(extract_segmentation_features(test_texts)))

    # ── 10. Radical features (optional) ───────────────────────────────────
    if use_radicals:
        from features.radicals import (
            extract_radical_features,
            radical_ngram_vectorizer,
            texts_to_radical_sequences,
        )

        header("Radical frequency distribution...")
        blocks_train.append(_sparse(extract_radical_features(train_texts)))
        blocks_val.append(_sparse(extract_radical_features(val_texts)))
        blocks_test.append(_sparse(extract_radical_features(test_texts)))

        header("Radical n-gram TF-IDF...")
        train_rad = texts_to_radical_sequences(train_texts, "    radical seq (train)")
        val_rad = texts_to_radical_sequences(val_texts, "    radical seq (val)")
        test_rad = texts_to_radical_sequences(test_texts, "    radical seq (test)")
        rad_vec = radical_ngram_vectorizer()
        blocks_train.append(rad_vec.fit_transform(train_rad))
        blocks_val.append(rad_vec.transform(val_rad))
        blocks_test.append(rad_vec.transform(test_rad))
        vectorizers["radical_ngram"] = rad_vec

    # ── 11. Dependency triples (optional) ─────────────────────────────────
    if use_dependency:
        from features.dependency import dep_triple_vectorizer, texts_to_dep_triples

        header("Dependency triple TF-IDF (spaCy)...")
        train_dep = texts_to_dep_triples(train_texts, "    dep parse (train)")
        val_dep = texts_to_dep_triples(val_texts, "    dep parse (val)")
        test_dep = texts_to_dep_triples(test_texts, "    dep parse (test)")
        dep_vec = dep_triple_vectorizer()
        blocks_train.append(dep_vec.fit_transform(train_dep))
        blocks_val.append(dep_vec.transform(val_dep))
        blocks_test.append(dep_vec.transform(test_dep))
        vectorizers["dep"] = dep_vec

    # ── Combine ───────────────────────────────────────────────────────────
    X_train = sp.hstack(blocks_train, format="csr")
    X_val = sp.hstack(blocks_val, format="csr")
    X_test = sp.hstack(blocks_test, format="csr")
>>>>>>> e689c55 (add more scripts and features)

    return X_train, X_val, X_test, vectorizers
