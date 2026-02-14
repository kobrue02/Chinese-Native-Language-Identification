from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import config


def build_svm(C: float = 1.0) -> LinearSVC:
    """Build a LinearSVC with balanced class weights."""
    return LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=10_000,
        random_state=config.RANDOM_SEED,
    )


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
