import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CV_N_REPEATS, CV_N_SPLITS, RANDOM_STATE, TOP_K_SHARE
from src.metrics import precision_at_k


def cross_validate_model(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
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
        n_splits: Number of folds per repeat.
        n_repeats: Number of times the split is repeated with a new shuffle.

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


def compare_pipelines(
    pipe_a: Pipeline,
    pipe_b: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, np.ndarray]:
    """Compare two pipelines fold by fold.

    Both pipelines are evaluated on the same folds, so the spread caused by
    folds differing in difficulty cancels out and only the effect of the change
    itself remains. A positive value means ``pipe_a`` scored higher.

    Args:
        pipe_a: Pipeline under test.
        pipe_b: Pipeline to compare against.
        X: Feature frame.
        y: Binary target.

    Returns:
        Per-fold differences, one array per metric.
    """
    cv_a = cross_validate_model(pipe_a, X, y)
    cv_b = cross_validate_model(pipe_b, X, y)

    return {name: cv_a[name] - cv_b[name] for name in cv_a}


def format_paired(diff: np.ndarray) -> str:
    """Format a paired difference as mean ± 2·SE.

    Four decimals: paired comparisons resolve differences down to ~0.001.
    """
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return f"{diff.mean():+.4f} ± {2 * se:.4f}"


def show_paired(diffs: dict[str, np.ndarray]) -> None:
    """Print one line per metric: paired difference and how often it improved.

    Ties count as non-improvements, so the counter is a lower bound.
    """
    for name, d in diffs.items():
        print(
            f"{name:15s} {format_paired(d)}   folds improved: {(d > 0).sum()} / {len(d)}"
        )


def show_absolute(res: dict[str, np.ndarray]) -> None:
    """Print one line per metric: mean ± std across folds.

    Three decimals: fold-to-fold spread is ~0.04, so the fourth is noise.
    """
    for name, v in res.items():
        print(f"{name:15s} {v.mean():.3f} ± {v.std(ddof=1):.3f}")
