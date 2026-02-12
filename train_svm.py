<<<<<<< HEAD
"""Train and evaluate the SVM baseline for NLI."""

=======
"""Train and evaluate traditional ML classifiers on NLI features.

Usage:
    python train_svm.py                          # LogisticRegression (fast, default)
    python train_svm.py --model sgd              # SGDClassifier (fastest)
    python train_svm.py --model svm              # LinearSVC (no grid search)
    python train_svm.py --model svm --gridsearch # LinearSVC with grid search (slow)
    python train_svm.py --model mlp              # MLPClassifier
    python train_svm.py --radicals               # + radical features
    python train_svm.py --dep                    # + dependency features (slow)
"""

import argparse
>>>>>>> e689c55 (add more scripts and features)
import time

from data_loader import load_and_split
from features import build_features
<<<<<<< HEAD
from models.svm import grid_search_svm
=======
from models.svm import build_classifier, grid_search_svm
>>>>>>> e689c55 (add more scripts and features)
from evaluate import evaluate_and_report


def main():
<<<<<<< HEAD
=======
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["logreg", "sgd", "svm", "mlp"],
        default="logreg",
        help="Classifier to use (default: logreg)",
    )
    parser.add_argument(
        "--gridsearch",
        action="store_true",
        help="Run grid search over C values (only for svm)",
    )
    parser.add_argument(
        "--dep",
        action="store_true",
        help="Include dependency triple features (requires zh_core_web_sm)",
    )
    parser.add_argument(
        "--radicals",
        action="store_true",
        help="Include radical features (requires data/radical_map.json; "
        "generate with: python -m features.radicals)",
    )
    args = parser.parse_args()

>>>>>>> e689c55 (add more scripts and features)
    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train, val, test, le = load_and_split()
    label_names = list(le.classes_)

    # ── Extract features ──────────────────────────────────────────────────
<<<<<<< HEAD
    print("Extracting features (this may take a few minutes)...")
    t0 = time.time()
    X_train, X_val, X_test, vectorizers = build_features(
        train["text"].tolist(),
        val["text"].tolist(),
        test["text"].tolist(),
=======
    print("Extracting features...")
    t0 = time.time()
    X_train, X_val, X_test, _ = build_features(
        train["text"].tolist(),
        val["text"].tolist(),
        test["text"].tolist(),
        use_dependency=args.dep,
        use_radicals=args.radicals,
>>>>>>> e689c55 (add more scripts and features)
    )
    print(f"Feature extraction took {time.time() - t0:.1f}s")
    print(f"Feature matrix shape: {X_train.shape}")

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

<<<<<<< HEAD
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
=======
    # ── Train ─────────────────────────────────────────────────────────────
    if args.model == "svm" and args.gridsearch:
        print("\nRunning SVM grid search...")
        gs = grid_search_svm(X_train, y_train)
        model = gs.best_estimator_
    else:
        print(f"\nTraining {args.model}...")
        model = build_classifier(args.model)
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"Training took {time.time() - t0:.1f}s")

    # ── Evaluate on validation set ────────────────────────────────────────
    y_val_pred = model.predict(X_val)
    tag = f"{args.model}_val"
    print(f"\n── Validation Set ({args.model}) ──")
    evaluate_and_report(y_val, y_val_pred, label_names, tag)

    # ── Evaluate on test set ──────────────────────────────────────────────
    y_test_pred = model.predict(X_test)
    tag = f"{args.model}_test"
    print(f"\n── Test Set ({args.model}) ──")
    evaluate_and_report(y_test, y_test_pred, label_names, tag)
>>>>>>> e689c55 (add more scripts and features)


if __name__ == "__main__":
    main()
