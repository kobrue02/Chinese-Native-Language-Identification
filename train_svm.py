"""Train and evaluate the SVM baseline for NLI."""

import time

from data_loader import load_and_split
from features import build_features
from models.svm import grid_search_svm
from evaluate import evaluate_and_report


def main():
    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train, val, test, le = load_and_split()
    label_names = list(le.classes_)

    # ── Extract features ──────────────────────────────────────────────────
    print("Extracting features (this may take a few minutes)...")
    t0 = time.time()
    X_train, X_val, X_test, vectorizers = build_features(
        train["text"].tolist(),
        val["text"].tolist(),
        test["text"].tolist(),
    )
    print(f"Feature extraction took {time.time() - t0:.1f}s")
    print(f"Feature matrix shape: {X_train.shape}")

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    # ── Grid search on train, pick best C ─────────────────────────────────
    print("\nRunning grid search...")
    gs = grid_search_svm(X_train, y_train)
    best_model = gs.best_estimator_

    # ── Evaluate on validation set ────────────────────────────────────────
    y_val_pred = best_model.predict(X_val)
    print("\n── Validation Set ──")
    evaluate_and_report(y_val, y_val_pred, label_names, "SVM_val")

    # ── Evaluate on test set ──────────────────────────────────────────────
    y_test_pred = best_model.predict(X_test)
    print("\n── Test Set ──")
    evaluate_and_report(y_test, y_test_pred, label_names, "SVM_test")


if __name__ == "__main__":
    main()
