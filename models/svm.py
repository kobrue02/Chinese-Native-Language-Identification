"""Traditional ML classifiers for NLI feature matrices."""

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import config


def build_classifier(name: str = "logreg"):
    """Build a classifier by name.

    Supported: logreg, sgd, svm, mlp.
    """
    seed = config.RANDOM_SEED
    if name == "logreg":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1_000,
            solver="saga",
            random_state=seed,
            n_jobs=-1,
            verbose=1,
        )
    elif name == "sgd":
        return SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            max_iter=1_000,
            random_state=seed,
            n_jobs=-1,
            verbose=1,
        )
    elif name == "svm":
        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=10_000,
            random_state=seed,
            verbose=1,
        )
    elif name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


def grid_search_svm(X_train, y_train) -> GridSearchCV:
    """Grid search over C values for LinearSVC."""
    param_grid = {"C": config.SVM_CONFIG["C_values"]}
    gs = GridSearchCV(
        LinearSVC(
            class_weight="balanced",
            max_iter=10_000,
            random_state=config.RANDOM_SEED,
        ),
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"Best C: {gs.best_params_['C']},  Best macro-F1 (CV): {gs.best_score_:.4f}")
    return gs
