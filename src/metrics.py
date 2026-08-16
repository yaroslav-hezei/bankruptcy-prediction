import numpy as np


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Share of positives among the k highest-scored samples.

    Ties at the cut-off are broken by input order: of the samples sharing the
    score at rank k, the earlier ones are selected. This matches how a queue of
    fixed capacity is filled in practice, at the cost of making the metric
    dependent on row order.

    Args:
        y_true: Binary labels, 0 or 1.
        y_score: Higher means higher predicted risk. Any monotone scale works;
            probabilities are not required.
        k: Number of top-ranked samples to evaluate.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer count, not a share, got {k}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score must have the same length, "
            f"got {len(y_true)} and {len(y_score)}"
        )
    if k > len(y_true):
        raise ValueError(f"k={k} exceeds the number of samples ({len(y_true)})")

    order = np.argsort(y_score)[::-1]
    n_positives = y_true[order[:k]].sum()

    return float(n_positives / k)
