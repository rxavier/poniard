import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from poniard import PoniardClassifier


@pytest.mark.parametrize(
    "X,preprocess,scaler,numeric_imputer,include_preprocessor",
    [
        (
            pd.DataFrame(
                {
                    "A": [4, 3, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                }
            ),
            True,
            None,
            None,
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                }
            ),
            True,
            "standard",
            "iterative",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                }
            ),
            True,
            "robust",
            "simple",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                }
            ),
            True,
            "minmax",
            None,
            True,
        ),
        (
            pd.DataFrame({"A": [4, 3, 1, -1, 0], "B": [-2, 1, 3, 7, 1]}),
            False,
            None,
            None,
            False,
        ),
    ],
)
def test_preprocessing_classifier(
    X, preprocess, scaler, numeric_imputer, include_preprocessor
):
    estimator = PoniardClassifier(
        estimators=[LogisticRegression()],
        preprocess=preprocess,
        scaler=scaler,
        numeric_imputer=numeric_imputer,
        cv=2,
        random_state=0,
    )
    y = [0, 1, 0, 1, 0]
    estimator.setup(X, y)
    estimator.fit()
    assert estimator.show_results().isna().sum().sum() == 0
    assert estimator.show_results().shape == (2, 12)
    assert isinstance(
        estimator.get_estimator(
            "LogisticRegression", include_preprocessor=include_preprocessor
        ),
        BaseEstimator,
    )
