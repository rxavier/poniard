import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor

from poniard import PoniardClassifier, PoniardRegressor
from poniard.preprocessing import PoniardPreprocessor


@pytest.mark.parametrize(
    "target,metrics,estimators,cv",
    [
        (np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]), None, None, None),
        (
            pd.Series(np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])),
            "accuracy",
            [LogisticRegression()],
            5,
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]).tolist(),
            ["accuracy", "roc_auc"],
            {"logreg": LogisticRegression(), "rf": RandomForestClassifier()},
            StratifiedKFold(n_splits=2),
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]),
            {"acc": make_scorer(accuracy_score), "roc": make_scorer(roc_auc_score)},
            [LogisticRegression(), RandomForestClassifier()],
            KFold(n_splits=3),
        ),
        (
            np.array([0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1]),
            None,
            [LogisticRegression(), RandomForestClassifier()],
            None,
        ),
    ],
)
def test_classifier_fit(target, metrics, estimators, cv):
    features = pd.DataFrame(np.random.normal(size=(len(target), 5)))
    features.columns = features.columns.astype(str)
    features["strings"] = np.random.choice(["a", "b", "c"], size=len(target))
    features["dates"] = pd.date_range("2020-01-01", periods=len(target))
    clf = PoniardClassifier(
        estimators=estimators, cv=cv, metrics=metrics, random_state=0
    )
    clf.setup(features, target)
    clf.fit(features, target)
    results = clf.get_results(return_train_scores=True)
    if not estimators:
        n_estimators = len(clf._default_estimators)
    else:
        n_estimators = len(estimators)
    if isinstance(metrics, str):
        n_metrics = 1
    else:
        n_metrics = len(clf.metrics)
    assert results.isna().sum().sum() == 0
    assert results.shape == (n_estimators + 1, n_metrics * 2 + 4)


@pytest.mark.parametrize(
    "target,metrics,estimators,cv",
    [
        (np.random.normal(size=(20,)), None, None, None),
        (
            pd.Series(np.random.normal(size=(20,))),
            "neg_mean_squared_error",
            [LinearRegression()],
            5,
        ),
        (
            np.random.normal(size=(20,)).tolist(),
            ["neg_mean_squared_error", "neg_mean_absolute_percentage_error"],
            {"linreg": LinearRegression(), "rf": RandomForestRegressor()},
            3,
        ),
        (
            np.random.normal(size=(20,)),
            {
                "mse": make_scorer(mean_squared_error, greater_is_better=False),
                "mape": make_scorer(
                    mean_absolute_percentage_error, greater_is_better=False
                ),
            },
            [LinearRegression(), RandomForestRegressor()],
            KFold(n_splits=3),
        ),
    ],
)
def test_regressor_fit(target, metrics, estimators, cv):
    features = pd.DataFrame(np.random.normal(size=(20, 5)))
    features.columns = features.columns.astype(str)
    features["strings"] = np.random.choice(["a", "b", "c"], size=len(target))
    features["dates"] = pd.date_range("2020-01-01", periods=len(target))
    clf = PoniardRegressor(
        estimators=estimators, cv=cv, metrics=metrics, random_state=0
    )
    clf.setup(features, target)
    clf.fit(features, target)
    results = clf.get_results(return_train_scores=True)
    if not estimators:
        n_estimators = len(clf._default_estimators)
    else:
        n_estimators = len(estimators)
    if isinstance(metrics, str):
        n_metrics = 1
    else:
        n_metrics = len(clf.metrics)
    assert results.isna().sum().sum() == 0
    assert results.shape == (n_estimators + 1, n_metrics * 2 + 4)


def test_multilabel_fit():
    import warnings
    X, y = make_multilabel_classification(n_samples=1000, n_classes=3, n_labels=3)
    clf = PoniardClassifier(
        estimators={
            "RF": OneVsRestClassifier(RandomForestClassifier()),
            "LR": OneVsRestClassifier(LogisticRegression()),
        },
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TargetEncoder is not supported")
        clf.setup(X, y)
        clf.fit(X, y)
    results = clf.get_results(return_train_scores=True)
    assert results.isna().sum().sum() == 0
    assert results.shape == (3, 14)


def test_multioutput_fit():
    import warnings
    X, y = make_regression(n_targets=3)
    clf = PoniardRegressor(
        estimators={
            "RF": MultiOutputRegressor(RandomForestRegressor()),
            "LR": MultiOutputRegressor(LinearRegression()),
        },
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TargetEncoder is not supported")
        clf.setup(X, y)
        clf.fit(X, y)
    results = clf.get_results(return_train_scores=True)
    assert results.isna().sum().sum() == 0
    assert results.shape == (3, 12)


def test_type_inference():
    x = pd.DataFrame(
        {
            "numeric": [float(i) for i in range(10)],
            "low_cardinality_str": ["a"] * 5 + ["b"] * 5,
            "low_cardinality_int": [1] * 10,
            "high_cardinality_str": [str(x) for x in range(10)],
            "high_cardinality_int": [x for x in range(10)],
            "datetime_H": pd.date_range("2020-01-01", freq="h", periods=10),
            "datetime_D": pd.date_range(
                "2020-01-01", freq="D", periods=10, tz="Europe/Moscow"
            ),
        }
    )
    # Add random nan to 10% per column: https://stackoverflow.com/a/61018279
    for col in x.columns:
        x.loc[x.sample(frac=0.1).index, col] = np.nan
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    preprocessor = PoniardPreprocessor(cardinality_threshold=0.3)
    clf = PoniardClassifier(
        estimators=[LogisticRegression()],
        cv=3,
        custom_preprocessor=preprocessor,
        random_state=0,
    )
    clf.setup(x, y)
    clf.fit(x, y)
    assert all(
        x in clf.feature_types["numeric"] for x in ["numeric", "high_cardinality_int"]
    )
    assert all(
        x in clf.feature_types["categorical_high"] for x in ["high_cardinality_str"]
    )
    assert all(
        x in clf.feature_types["categorical_low"]
        for x in ["low_cardinality_str", "low_cardinality_int"]
    )
    assert all(x in clf.feature_types["datetime"] for x in ["datetime_H", "datetime_D"])
