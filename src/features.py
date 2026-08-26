import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def add_negative_equity_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean flag marking rows where equity < 0."""
    result = df.copy()
    result["has_negative_equity"] = df["Attr10"] < 0
    return result


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Drop features that duplicate a column standing to their left.

    A column is dropped when its absolute Spearman correlation with any column to its
    left exceeds the threshold. Members of a correlated group are interchangeable, so the
    choice of survivor is arbitrary by design: ranking them by separability would read
    noise and would tie the feature set to the target.

    The scan is transitive — with A-B and B-C above the threshold, C is dropped even when
    A and C are not correlated.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "CorrelationSelector":
        """Record which columns to drop, using the training rows only."""
        corr = X.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        self.to_drop_ = [c for c in upper.columns if upper[c].gt(self.threshold).any()]
        self.kept_ = [c for c in X.columns if c not in self.to_drop_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the retained columns, in the order seen at fit time."""
        check_is_fitted(self)
        return X[self.kept_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the retained columns, for pandas output downstream."""
        check_is_fitted(self)
        return np.asarray(self.kept_, dtype=object)
