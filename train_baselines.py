"""Compute baseline classifiers for NLI.

Baselines:
  - Majority class (most frequent label)
  - Stratified random (sample according to training distribution)
  - Uniform random (sample uniformly from all classes)

Usage:
    python train_baselines.py
"""

import numpy as np
from collections import Counter

import config
from data_loader import load_and_split
from evaluate import evaluate_and_report


def majority_predict(y_train: np.ndarray, n: int) -> np.ndarray:
    """Predict the most frequent training label for all samples."""
    majority = Counter(y_train).most_common(1)[0][0]
    return np.full(n, majority)


def stratified_random_predict(
    y_train: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Predict by sampling from the training label distribution."""
    rng = np.random.RandomState(seed)
    counts = Counter(y_train)
    labels = np.array(list(counts.keys()))
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    return rng.choice(labels, size=n, p=probs)


def uniform_random_predict(
    y_train: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Predict by sampling uniformly from all training classes."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    return rng.choice(classes, size=n)


def main():
    print("Loading data...")
    train, val, test, le = load_and_split()
    label_names = list(le.classes_)

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    baselines = {
        "majority": lambda n: majority_predict(y_train, n),
        "stratified_random": lambda n: stratified_random_predict(
            y_train, n, config.RANDOM_SEED
        ),
        "uniform_random": lambda n: uniform_random_predict(
            y_train, n, config.RANDOM_SEED
        ),
    }

    for name, predict_fn in baselines.items():
        print(f"\n{'─' * 60}")
        print(f"Baseline: {name}")
        print(f"{'─' * 60}")

        y_val_pred = predict_fn(len(y_val))
        print("\n── Validation Set ──")
        evaluate_and_report(y_val, y_val_pred, label_names, f"{name}_val")

        y_test_pred = predict_fn(len(y_test))
        print("\n── Test Set ──")
        evaluate_and_report(y_test, y_test_pred, label_names, f"{name}_test")


if __name__ == "__main__":
    main()
