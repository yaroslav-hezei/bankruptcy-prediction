import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

Z_PRIME_WEIGHTS = {
    "Attr3": 0.717,
    "Attr6": 0.847,
    "Attr7": 3.107,
    "Attr8": 0.420,
    "Attr9": 0.998,
}


class AltmanZScore(BaseEstimator, ClassifierMixin):
    """Altman Z'-score (1983) for private firms, wrapped as a classifier.

    The weights are fixed at their published values and nothing is learned from
    the data — fit only records the class labels. The score is negated so that
    higher means higher risk, matching the direction the ranking metrics expect,
    and squashed through a sigmoid to satisfy the predict_proba contract. The
    resulting numbers are not calibrated probabilities and should only be used
    for ranking.

    Missing values are not handled here: the formula propagates NaN. Pair the
    estimator with an imputer in a Pipeline.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        columns = list(Z_PRIME_WEIGHTS)
        values = X[columns].to_numpy()
        weight = np.array(list(Z_PRIME_WEIGHTS.values()))
        z = values @ weight
        risk = -z
        p = 1 / (1 + np.exp(-np.clip(risk, -500, 500)))
        return np.column_stack([1 - p, p])
