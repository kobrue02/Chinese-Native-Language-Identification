import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


def load_corpus() -> pd.DataFrame:
    """Read index.csv and corresponding text files into a DataFrame."""
    df = pd.read_csv(
        config.INDEX_CSV,
        header=None,
        names=["doc_id", "context", "native_language", "gender"],
    )
    # Load text for each document
    texts = []
    for doc_id in df["doc_id"]:
        path = config.DATA_DIR / f"{doc_id}.txt"
        texts.append(path.read_text(encoding="utf-8").strip())
    df["text"] = texts
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Fit a LabelEncoder on native_language and add a 'label' column."""
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df["native_language"])
    return df, le


def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test with stratification on native_language.

    For classes with fewer than 3 samples, all samples go to train to
    avoid errors in stratified splitting.
    """
    rng = np.random.RandomState(config.RANDOM_SEED)

    # Drop any rows with missing language
    df = df.dropna(subset=["native_language"])

    counts = df["native_language"].value_counts()
    rare_langs = counts[counts < 3].index
    rare_mask = df["native_language"].isin(rare_langs)
    df_rare = df[rare_mask]
    
    df_main = df[~rare_mask]
    # filter out any language that appears only once, since stratified splitting would fail
    df_main = df_main[df_main["native_language"].map(df_main["native_language"].value_counts()) > 1]

    # First split: train vs (val+test)
    val_test_ratio = config.VAL_RATIO + config.TEST_RATIO
    df_train, df_valtest = train_test_split(
        df_main,
        test_size=val_test_ratio,
        random_state=config.RANDOM_SEED,
        stratify=df_main["native_language"],
    )

    # Second split: val vs test (equal halves of the val+test portion)
    relative_test = config.TEST_RATIO / val_test_ratio
    df_valtest = df_valtest[df_valtest["native_language"].map(df_valtest["native_language"].value_counts()) > 1]
    df_val, df_test = train_test_split(
        df_valtest,
        test_size=relative_test,
        random_state=config.RANDOM_SEED,
        stratify=df_valtest["native_language"],
    )

    # Add rare-class samples to train
    df_train = pd.concat([df_train, df_rare], ignore_index=True)

    # Shuffle train split
    df_train = df_train.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(
        drop=True
    )
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_val, df_test


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Convenience: load corpus, encode labels, split."""
    df = load_corpus()
    df, le = encode_labels(df)
    train, val, test = stratified_split(df)
    return train, val, test, le


if __name__ == "__main__":
    train, val, test, le = load_and_split()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Classes: {len(le.classes_)}")
    print(f"Labels: {list(le.classes_)}")
