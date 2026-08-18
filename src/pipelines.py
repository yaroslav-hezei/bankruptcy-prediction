from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from src.config import MISSING_INDICATOR_COLS, RANDOM_STATE


def make_boosting_pipeline(model: BaseEstimator, feature_cols: list[str]) -> Pipeline:
    """Build a pipeline for gradient boosting.

    Features pass through untouched: the model splits on order rather than
    magnitude and routes NaN natively. Only the missingness flags are added,
    since a gap encodes a financial state the raw column cannot express.

    Args:
        model: Unfitted estimator exposing ``predict_proba``.
        feature_cols: Feature columns, in the order they reach the model.

    Returns:
        Pipeline with steps ``prep`` and ``model``, producing pandas output.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("flags", MissingIndicator(features="all"), MISSING_INDICATOR_COLS),
            ("raw", "passthrough", feature_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    pipe.set_output(transform="pandas")
    return pipe


def make_linear_pipeline(model: BaseEstimator, feature_cols: list[str]) -> Pipeline:
    """Build a pipeline for a linear model.

    Median imputation is required — the estimator rejects NaN — and destroys the
    information that a value was missing, which the flags carry instead. The
    quantile transform replaces values by their rank: tails here reach hundreds
    of times the 99th percentile, and any linear rescaling would leave them
    dominating the coefficients.

    Args:
        model: Unfitted estimator exposing ``predict_proba``.
        feature_cols: Feature columns, in the order they reach the model.

    Returns:
        Pipeline with steps ``prep`` and ``model``, producing pandas output.
    """

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=500,
                    output_distribution="normal",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("flags", MissingIndicator(features="all"), MISSING_INDICATOR_COLS),
            ("num", numeric_pipe, feature_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    pipe.set_output(transform="pandas")
    return pipe
