import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

from poniard import PoniardClassifier, PoniardRegressor


@pytest.mark.parametrize(
    "X,preprocess,scaler,numeric_imputer,high_cardinality_encoder,include_preprocessor",
    [
        (
            pd.DataFrame(
                {
                    "A": [4, 3, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="M", periods=5),
                }
            ),
            True,
            None,
            None,
            "target",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="H", periods=5),
                }
            ),
            True,
            "standard",
            "iterative",
            "ordinal",
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="Y", periods=5),
                }
            ),
            True,
            "robust",
            "simple",
            None,
            True,
        ),
        (
            pd.DataFrame(
                {
                    "A": [4, 200, 1, -1, np.nan],
                    "B": [-2, np.nan, 3, 7, 1],
                    "C": list("abcde"),
                    "D": pd.date_range("2020-01-01", freq="MS", periods=5),
                }
            ),
            True,
            "minmax",
            None,
            "target",
            True,
        ),
        (
            pd.DataFrame({"A": [4, 3, 1, -1, 0], "B": [-2, 1, 3, 7, 1]}),
            False,
            None,
            None,
            "ordinal",
            False,
        ),
    ],
)
def test_preprocessing_classifier(
    X,
    preprocess,
    scaler,
    numeric_imputer,
    high_cardinality_encoder,
    include_preprocessor,
):
    estimator = PoniardClassifier(
        estimators=[LogisticRegression()],
        preprocess=preprocess,
        scaler=scaler,
        numeric_imputer=numeric_imputer,
        high_cardinality_encoder=high_cardinality_encoder,
        cv=2,
        random_state=0,
    )
    y = [0, 1, 0, 1, 0]
    estimator.setup(X, y)
    estimator.fit()
    assert estimator.get_results().isna().sum().sum() == 0
    assert estimator.get_results(return_train_scores=True).shape == (2, 12)
    assert isinstance(
        estimator.get_estimator(
            "LogisticRegression", include_preprocessor=include_preprocessor
        ),
        BaseEstimator,
    )


@pytest.mark.parametrize(
    "new_step,position,existing_step",
    [
        (SelectKBest(f_regression, k=2), 0, None),
        (
            make_pipeline(SimpleImputer(), SelectKBest(f_regression, k=2)),
            "start",
            StandardScaler(),
        ),
        (
            make_pipeline(SimpleImputer(), SelectKBest(f_regression, k=2)),
            "end",
            make_pipeline(SimpleImputer(), StandardScaler()),
        ),
    ],
)
def test_add_step(new_step, position, existing_step):
    X = pd.DataFrame(
        {
            "A": [4, 3, 1, -1, np.nan],
            "B": [-2, np.nan, 3, 7, 1],
            "C": list("abcde"),
            "D": pd.date_range("2020-01-01", freq="M", periods=5),
        }
    )
    y = np.random.uniform(0, 1, size=5)
    reg = PoniardRegressor(custom_preprocessor=existing_step).setup(X, y)
    reg.add_preprocessing_step(new_step, position)
    assert isinstance(reg.preprocessor, Pipeline)
