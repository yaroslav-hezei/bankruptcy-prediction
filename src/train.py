import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import CV_N_REPEATS, CV_N_SPLITS, RANDOM_STATE, TOP_K_SHARE
from src.metrics import precision_at_k


def cross_validate_model(
    pipe,
    X,
    y,
    k_share: float = TOP_K_SHARE,
    n_splits: int = CV_N_SPLITS,
    n_repeats: int = CV_N_REPEATS,
) -> dict[str, np.ndarray]:
    """Evaluate a pipeline with repeated stratified cross-validation.

    Returns per-fold values rather than aggregates: the spread across folds is
    part of the result, and mean alone would hide it. The splitter is seeded, so
    every model evaluated here sees the same folds and results are comparable.

    Args:
        pipe: Unfitted estimator. Cloned before each fit.
        X: Feature matrix as a DataFrame.
        y: Binary labels as a Series.
        k_share: Size of the review queue as a share of the fold.

    Returns:
        Arrays of per-fold scores, keyed by metric name.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )

    pr_auc_scores = []
    precision_k_scores = []

    for train_idx, val_idx in splitter.split(X, y):
        model = clone(pipe)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = model.predict_proba(X.iloc[val_idx])[:, 1]

        y_val = y.iloc[val_idx]
        pr_auc = average_precision_score(y_val, scores)

        k = int(len(y_val) * k_share)
        precision_k = precision_at_k(y_val, scores, k)

        pr_auc_scores.append(pr_auc)
        precision_k_scores.append(precision_k)

    return {
        "pr_auc": np.array(pr_auc_scores),
        "precision_at_k": np.array(precision_k_scores),
    }
